/**
 * dashboard.js — Agent simulation viewer.
 * Runs a full episode against the API and displays each step.
 */

const Dashboard = (() => {
  // ── DOM refs ──
  const taskSelect  = () => document.getElementById('dashTaskSelect');
  const runBtn      = () => document.getElementById('dashRunBtn');
  const resetBtn    = () => document.getElementById('dashResetBtn');
  const codeViewer  = () => document.getElementById('dashCodeViewer');
  const taskBadge   = () => document.getElementById('dashTaskBadge');
  const stepBadge   = () => document.getElementById('dashStepBadge');
  const logContainer= () => document.getElementById('dashLog');
  const gradingPanel= () => document.getElementById('dashGradingPanel');
  const matchedVal  = () => document.getElementById('dashMatchedVal');
  const totalGtVal  = () => document.getElementById('dashTotalGtVal');
  const submittedVal= () => document.getElementById('dashSubmittedVal');
  const falsePosEl  = () => document.getElementById('dashFalsePos');
  const rewardBar   = () => document.getElementById('dashRewardBar');
  const rewardLabel = () => document.getElementById('dashRewardLabel');

  let running = false;

  // ── Render code snippet ──
  function renderCode(snippet) {
    const viewer = codeViewer();
    viewer.innerHTML = '';
    const lines = snippet.split('\n');
    lines.forEach((text, idx) => {
      const lineEl = document.createElement('div');
      lineEl.className = 'code-line';
      lineEl.dataset.line = idx + 1;

      const numEl = document.createElement('span');
      numEl.className = 'code-line-number';
      numEl.textContent = idx + 1;

      const contentEl = document.createElement('span');
      contentEl.className = 'code-line-content';
      contentEl.textContent = text;

      const marker = document.createElement('span');
      marker.className = 'code-line-marker';

      lineEl.appendChild(marker);
      lineEl.appendChild(numEl);
      lineEl.appendChild(contentEl);
      viewer.appendChild(lineEl);
    });
  }

  // ── Highlight lines where issues were found ──
  function highlightLine(lineNum, correct) {
    const viewer = codeViewer();
    const lineEl = viewer.querySelector(`[data-line="${lineNum}"]`);
    if (lineEl) {
      lineEl.classList.add(correct ? 'flagged-correct' : 'flagged-wrong');
      const marker = lineEl.querySelector('.code-line-marker');
      if (marker) marker.textContent = correct ? '+' : '!';
    }
  }

  // ── Add log entry ──
  function addLog(step, icon, text, reward) {
    const container = logContainer();
    // Remove empty state if present
    const empty = container.querySelector('.empty-state');
    if (empty) empty.remove();

    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const stepEl = document.createElement('span');
    stepEl.className = 'log-step';
    stepEl.textContent = `#${step}`;

    const iconEl = document.createElement('span');
    iconEl.className = 'log-icon';
    iconEl.textContent = icon;

    const textEl = document.createElement('span');
    textEl.className = 'log-text';
    textEl.textContent = text;

    const rewardEl = document.createElement('span');
    const rClass = reward > 0 ? 'positive' : reward < 0 ? 'negative' : 'neutral';
    rewardEl.className = `log-reward ${rClass}`;
    rewardEl.textContent = reward > 0 ? `+${reward.toFixed(3)}` : reward.toFixed(3);

    entry.appendChild(stepEl);
    entry.appendChild(iconEl);
    entry.appendChild(textEl);
    entry.appendChild(rewardEl);
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
  }

  // ── Update grading panel ──
  function updateGrading(matched, totalGt, submitted, falsePos, totalReward) {
    gradingPanel().style.display = 'block';
    matchedVal().textContent = matched;
    totalGtVal().textContent = totalGt;
    submittedVal().textContent = submitted;
    falsePosEl().textContent = falsePos;

    const pct = Math.max(0, Math.min(100, totalReward * 100));
    const bar = rewardBar();
    bar.style.width = pct + '%';
    bar.classList.toggle('negative', totalReward < 0);
    rewardLabel().textContent = totalReward.toFixed(3);
  }

  // ── Simulated agent strategy ──
  // Mimics what inference.py does: review -> hint (if medium/hard) -> review again -> done
  async function runSimulation() {
    if (running) return;
    running = true;
    runBtn().disabled = true;
    resetBtn().disabled = true;

    const taskId = taskSelect().value;
    const diff = taskId.split('_')[0];

    // Reset log
    logContainer().innerHTML = '';
    gradingPanel().style.display = 'none';
    taskBadge().textContent = taskId;

    // Step 0: Reset
    let obs;
    try {
      obs = await API.reset(taskId);
    } catch (e) {
      addLog(0, 'X', `Reset failed: ${e.message}`, 0);
      running = false;
      runBtn().disabled = false;
      return;
    }

    const episodeId = obs.episode_id;
    renderCode(obs.code_snippet);
    addLog(0, '>', `Episode started: ${taskId}`, 0);
    stepBadge().textContent = 'Step 0';

    let stepNum = 0;
    let totalReward = 0;
    let totalFindings = 0;
    let matched = 0;

    // Helper: sleep for visual effect
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    // Phase 1: Initial review — submit some "agent-generated" findings
    // We'll read the code and simulate finding some issues
    const codeLines = obs.code_snippet.split('\n');
    const initialFindings = generateSimulatedFindings(codeLines, diff, 'initial');

    if (initialFindings.length > 0) {
      await sleep(800);
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      let resp;
      try {
        resp = await API.submitFindings(episodeId, initialFindings, false);
      } catch (e) {
        addLog(stepNum, 'X', `Step failed: ${e.message}`, 0);
        running = false;
        runBtn().disabled = false;
        resetBtn().disabled = false;
        return;
      }
      totalReward = resp.reward || 0;
      const feedbackText = resp.feedback || 'Findings submitted';
      addLog(stepNum, '>', `Submitted ${initialFindings.length} finding(s) — ${feedbackText}`, resp.reward || 0);
      initialFindings.forEach(f => highlightLine(f.line_number, true));
      totalFindings += initialFindings.length;
    }

    // Phase 2: Request hint (for medium/hard)
    if (diff !== 'easy') {
      await sleep(600);
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const hintResp = await API.requestHint(episodeId);
        totalReward = hintResp.reward || totalReward;
        const hintText = hintResp.hint || 'No hint available';
        addLog(stepNum, '?', `Hint: ${hintText}`, -0.05);
      } catch (e) {
        addLog(stepNum, '!', `Hint failed: ${e.message}`, 0);
      }
    }

    // Phase 3: Request analysis (for hard)
    if (diff === 'hard') {
      await sleep(600);
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const analysisResp = await API.requestAnalysis(episodeId);
        totalReward = analysisResp.reward || totalReward;
        const analysisText = analysisResp.analysis_result || 'No analysis available';
        addLog(stepNum, '@', `Analysis: ${analysisText.substring(0, 120)}...`, -0.10);
      } catch (e) {
        addLog(stepNum, '!', `Analysis failed: ${e.message}`, 0);
      }
    }

    // Phase 4: Submit more findings based on hints
    const extraFindings = generateSimulatedFindings(codeLines, diff, 'refined');
    if (extraFindings.length > 0) {
      await sleep(800);
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const resp = await API.submitFindings(episodeId, extraFindings, false);
        totalReward = resp.reward || totalReward;
        addLog(stepNum, '>', `Submitted ${extraFindings.length} more finding(s) — ${resp.feedback || ''}`, resp.reward || 0);
        extraFindings.forEach(f => highlightLine(f.line_number, true));
        totalFindings += extraFindings.length;
      } catch (e) {
        addLog(stepNum, '!', `Step failed: ${e.message}`, 0);
      }
    }

    // Phase 5: Done
    await sleep(500);
    stepNum++;
    stepBadge().textContent = `Step ${stepNum}`;
    try {
      const doneResp = await API.sendDone(episodeId);
      totalReward = doneResp.reward || totalReward;
      addLog(stepNum, '#', `Episode complete. Final reward: ${totalReward.toFixed(3)}`, totalReward);
    } catch (e) {
      addLog(stepNum, '!', `Done failed: ${e.message}`, 0);
    }

    // Calculate rough grading stats
    const allFindings = initialFindings.length + extraFindings.length;
    const estimatedGt = diff === 'easy' ? 5 : diff === 'medium' ? 5 : 8;
    const approxMatched = Math.min(allFindings, Math.round(estimatedGt * Math.max(0, totalReward + 0.15)));
    const falsePos = Math.max(0, allFindings - approxMatched);

    updateGrading(approxMatched, estimatedGt, allFindings, falsePos, totalReward);

    running = false;
    runBtn().disabled = false;
    resetBtn().disabled = false;
  }

  // ── Simple heuristic finding generator ──
  // Scans code lines for common patterns to simulate what an agent might find
  function generateSimulatedFindings(lines, difficulty, phase) {
    const findings = [];

    if (phase === 'initial') {
      lines.forEach((line, idx) => {
        const lineNum = idx + 1;
        const trimmed = line.trim();

        // Detect unused imports (easy/common pattern)
        if (trimmed.startsWith('import ') && !trimmed.includes('from')) {
          const mod = trimmed.replace('import ', '').split(',')[0].trim();
          // Simple heuristic: small modules often unused
          if (['sys', 'json', 'io', 'time', 're', 'collections'].includes(mod)) {
            findings.push({
              line_number: lineNum,
              issue_type: 'style',
              severity: 'low',
              description: `Potentially unused import: '${mod}'`,
            });
          }
        }

        // Detect naming violations
        if (trimmed.startsWith('def ') && /def [A-Z]/.test(trimmed)) {
          const name = trimmed.match(/def (\w+)/)?.[1];
          if (name) {
            findings.push({
              line_number: lineNum,
              issue_type: 'style',
              severity: 'low',
              description: `Function '${name}' should use snake_case`,
            });
          }
        }

        // Detect SQL injection
        if (trimmed.includes('f"SELECT') || trimmed.includes("f'SELECT") ||
            trimmed.includes('f"UPDATE') || trimmed.includes("f'UPDATE") ||
            (trimmed.includes('.execute(') && trimmed.includes('+'))) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'critical',
            description: 'SQL injection: unsanitised input in query',
          });
        }

        // Detect os.system / command injection
        if (trimmed.includes('os.system(') || (trimmed.includes('shell=True') && trimmed.includes('subprocess'))) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'critical',
            description: 'Command injection: shell execution with user input',
          });
        }

        // Detect hardcoded secrets
        if (/^[A-Z_]+(SECRET|KEY|PASSWORD|TOKEN)\s*=\s*["']/.test(trimmed) ||
            /^(SECRET|KEY|PASSWORD|DB_PASSWORD|JWT_SECRET|WEBHOOK_SECRET|API_SECRET|SMTP_PASSWORD)\s*=/.test(trimmed)) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'critical',
            description: 'Hardcoded secret/credential in source code',
          });
        }

        // Detect eval()
        if (trimmed.includes('eval(') && !trimmed.startsWith('#')) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'critical',
            description: 'Code injection via eval() on untrusted input',
          });
        }

        // Detect pickle.loads
        if (trimmed.includes('pickle.loads(')) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'high',
            description: 'Insecure deserialization via pickle.loads',
          });
        }
      });
    }

    if (phase === 'refined') {
      lines.forEach((line, idx) => {
        const lineNum = idx + 1;
        const trimmed = line.trim();

        // Division by zero risk
        if (trimmed.includes('/ len(') && !trimmed.includes('if ')) {
          findings.push({
            line_number: lineNum,
            issue_type: 'bug',
            severity: 'high',
            description: 'Potential ZeroDivisionError when collection is empty',
          });
        }

        // yaml.load without safe_load
        if (trimmed.includes('yaml.load(') && !trimmed.includes('safe_load')) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'critical',
            description: 'Unsafe YAML loading; use yaml.safe_load()',
          });
        }

        // MD5 / SHA1 for passwords
        if (trimmed.includes('hashlib.md5(') || trimmed.includes('hashlib.sha1(')) {
          findings.push({
            line_number: lineNum,
            issue_type: 'security',
            severity: 'high',
            description: 'Weak hash algorithm for password hashing',
          });
        }

        // Class naming
        if (trimmed.startsWith('class ') && /class [a-z]/.test(trimmed)) {
          const name = trimmed.match(/class (\w+)/)?.[1];
          if (name) {
            findings.push({
              line_number: lineNum,
              issue_type: 'style',
              severity: 'medium',
              description: `Class '${name}' should use PascalCase`,
            });
          }
        }
      });
    }

    // Limit to avoid too many findings
    return findings.slice(0, phase === 'initial' ? 6 : 4);
  }

  // ── Reset ──
  function reset() {
    if (running) return;
    logContainer().innerHTML = '<div class="empty-state" style="padding:24px;"><div class="empty-icon">&#128203;</div>Episode steps will appear here</div>';
    codeViewer().innerHTML = '<div class="empty-state"><div class="empty-icon">&#128196;</div>Select a task and run the simulation</div>';
    gradingPanel().style.display = 'none';
    stepBadge().textContent = 'Step 0';
    taskBadge().textContent = '--';
  }

  // ── Event listeners ──
  function init() {
    runBtn().addEventListener('click', runSimulation);
    resetBtn().addEventListener('click', reset);
  }

  // Auto-init when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { runSimulation, reset };
})();
