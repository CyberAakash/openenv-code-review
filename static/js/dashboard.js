/**
 * dashboard.js — Agent simulation viewer with configurable heuristics + LLM mode.
 *
 * Default mode: Smart randomized heuristics — each run produces different results.
 * Optional mode: LLM-powered (calls an OpenAI-compatible API from the browser).
 *
 * Settings are persisted in localStorage.
 */

const Dashboard = (() => {
  // ── DOM refs ──
  const $ = id => document.getElementById(id);
  const taskSelect   = () => $('dashTaskSelect');
  const runBtn       = () => $('dashRunBtn');
  const resetBtn     = () => $('dashResetBtn');
  const codeViewer   = () => $('dashCodeViewer');
  const taskBadge    = () => $('dashTaskBadge');
  const stepBadge    = () => $('dashStepBadge');
  const logContainer = () => $('dashLog');
  const gradingPanel = () => $('dashGradingPanel');
  const matchedVal   = () => $('dashMatchedVal');
  const totalGtVal   = () => $('dashTotalGtVal');
  const submittedVal = () => $('dashSubmittedVal');
  const falsePosEl   = () => $('dashFalsePos');
  const rewardBar    = () => $('dashRewardBar');
  const rewardLabel  = () => $('dashRewardLabel');

  // Settings DOM
  const settingsToggle = () => $('dashSettingsToggle');
  const settingsPanel  = () => $('dashSettingsPanel');
  const modeBadge      = () => $('dashModeBadge');
  const randomnessSlider = () => $('dashRandomness');
  const randomnessLabel  = () => $('dashRandomnessLabel');
  const chkFalsePos    = () => $('dashFalsePositives');
  const chkHints       = () => $('dashUseHints');
  const chkAnalysis    = () => $('dashUseAnalysis');
  const chkUseLLM      = () => $('dashUseLLM');
  const llmConfig      = () => $('dashLLMConfig');
  const llmUrl         = () => $('dashLLMUrl');
  const llmKey         = () => $('dashLLMKey');
  const llmModel       = () => $('dashLLMModel');
  const llmStatus      = () => $('dashLLMStatus');

  let running = false;

  // ══════════════════════════════════════════════════════════════════
  //  SETTINGS STATE — persist to localStorage
  // ══════════════════════════════════════════════════════════════════
  const STORAGE_KEY = 'dashSimSettings';
  const RANDOMNESS_LABELS = ['Conservative', 'Balanced', 'Chaotic'];

  function defaultSettings() {
    return {
      randomness: 1,         // 0=conservative, 1=balanced, 2=chaotic
      falsePositives: true,
      useHints: true,
      useAnalysis: true,
      useLLM: false,
      llmUrl: '',
      llmKey: '',
      llmModel: 'gpt-4o-mini',
    };
  }

  function loadSettings() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return { ...defaultSettings(), ...JSON.parse(raw) };
    } catch { /* ignore */ }
    return defaultSettings();
  }

  function saveSettings(s) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  }

  function readSettingsFromUI() {
    return {
      randomness:     parseInt(randomnessSlider().value, 10),
      falsePositives: chkFalsePos().checked,
      useHints:       chkHints().checked,
      useAnalysis:    chkAnalysis().checked,
      useLLM:         chkUseLLM().checked,
      llmUrl:         llmUrl().value.trim(),
      llmKey:         llmKey().value.trim(),
      llmModel:       llmModel().value.trim() || 'gpt-4o-mini',
    };
  }

  function applySettingsToUI(s) {
    randomnessSlider().value = s.randomness;
    randomnessLabel().textContent = RANDOMNESS_LABELS[s.randomness] || 'Balanced';
    chkFalsePos().checked = s.falsePositives;
    chkHints().checked    = s.useHints;
    chkAnalysis().checked = s.useAnalysis;
    chkUseLLM().checked   = s.useLLM;
    llmUrl().value   = s.llmUrl;
    llmKey().value   = s.llmKey;
    llmModel().value = s.llmModel;
    updateLLMVisibility(s.useLLM);
    updateModeBadge(s.useLLM);
  }

  function updateLLMVisibility(enabled) {
    const cfg = llmConfig();
    if (cfg) cfg.classList.toggle('visible', enabled);
  }

  function updateModeBadge(isLLM) {
    const badge = modeBadge();
    if (!badge) return;
    badge.className = 'mode-badge ' + (isLLM ? 'llm' : 'heuristic');
    badge.textContent = isLLM ? 'LLM' : 'Heuristic';
  }

  // ══════════════════════════════════════════════════════════════════
  //  RENDER HELPERS
  // ══════════════════════════════════════════════════════════════════
  function renderCode(snippet) {
    const viewer = codeViewer();
    viewer.innerHTML = '';
    snippet.split('\n').forEach((text, idx) => {
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

  function highlightLine(lineNum, correct) {
    const viewer = codeViewer();
    const lineEl = viewer.querySelector(`[data-line="${lineNum}"]`);
    if (lineEl) {
      lineEl.classList.add(correct ? 'flagged-correct' : 'flagged-wrong');
      const marker = lineEl.querySelector('.code-line-marker');
      if (marker) marker.textContent = correct ? '+' : '!';
    }
  }

  function addLog(step, icon, text, reward) {
    const container = logContainer();
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

  // ══════════════════════════════════════════════════════════════════
  //  RANDOM UTILITIES
  // ══════════════════════════════════════════════════════════════════
  function randInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function randFloat(min, max) {
    return Math.random() * (max - min) + min;
  }

  function shuffle(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function randomSubset(arr, keepRate) {
    return arr.filter(() => Math.random() < keepRate);
  }

  // ══════════════════════════════════════════════════════════════════
  //  HEURISTIC FINDING DETECTOR (deterministic base layer)
  // ══════════════════════════════════════════════════════════════════
  function detectAllFindings(lines) {
    const findings = [];

    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      const trimmed = line.trim();

      // ── Style: unused imports ──
      if (trimmed.startsWith('import ') && !trimmed.includes('from')) {
        const mod = trimmed.replace('import ', '').split(',')[0].trim();
        if (['sys', 'json', 'io', 'time', 're', 'collections', 'os', 'tempfile', 'hashlib'].includes(mod)) {
          findings.push({
            line_number: lineNum,
            issue_type: 'style',
            severity: 'low',
            description: `Potentially unused import: '${mod}'`,
          });
        }
      }

      // ── Style: function naming ──
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

      // ── Style: class naming ──
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

      // ── Style: constant naming ──
      if (/^[a-z_]+\s*=\s*\d+/.test(trimmed) && !trimmed.startsWith('def ') && !trimmed.startsWith('class ') && !trimmed.includes('self.')) {
        const varName = trimmed.match(/^([a-z_]+)\s*=/)?.[1];
        if (varName && varName.length <= 15 && /^[a-z]/.test(varName) && trimmed.match(/=\s*\d+$/)) {
          findings.push({
            line_number: lineNum,
            issue_type: 'style',
            severity: 'low',
            description: `Consider using UPPER_CASE for constant '${varName}'`,
          });
        }
      }

      // ── Security: SQL injection ──
      if (trimmed.includes('f"SELECT') || trimmed.includes("f'SELECT") ||
          trimmed.includes('f"UPDATE') || trimmed.includes("f'UPDATE") ||
          trimmed.includes('f"INSERT') || trimmed.includes("f'INSERT") ||
          trimmed.includes('f"DELETE') || trimmed.includes("f'DELETE") ||
          (trimmed.includes('.execute(') && (trimmed.includes('+') || trimmed.includes('f"') || trimmed.includes("f'")))) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'SQL injection: unsanitised input in query string',
        });
      }

      // ── Security: command injection ──
      if (trimmed.includes('os.system(') ||
          (trimmed.includes('subprocess') && trimmed.includes('shell=True'))) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'Command injection: shell execution with user input',
        });
      }

      // ── Security: hardcoded secrets ──
      if (/^[A-Z_]*(SECRET|KEY|PASSWORD|TOKEN)\s*=\s*["']/.test(trimmed) ||
          /^(SECRET|KEY|PASSWORD|DB_PASSWORD|JWT_SECRET|WEBHOOK_SECRET|API_SECRET|SMTP_PASSWORD)\s*=/.test(trimmed)) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'Hardcoded secret/credential in source code',
        });
      }

      // ── Security: eval() ──
      if (trimmed.includes('eval(') && !trimmed.startsWith('#')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'Code injection via eval() on untrusted input',
        });
      }

      // ── Security: pickle.loads ──
      if (trimmed.includes('pickle.loads(')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'high',
          description: 'Insecure deserialization via pickle.loads',
        });
      }

      // ── Security: yaml.load without safe_load ──
      if (trimmed.includes('yaml.load(') && !trimmed.includes('safe_load')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'Unsafe YAML loading; use yaml.safe_load()',
        });
      }

      // ── Security: MD5 / SHA1 for passwords ──
      if (trimmed.includes('hashlib.md5(') || trimmed.includes('hashlib.sha1(')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'high',
          description: 'Weak hash algorithm; avoid MD5/SHA1 for password hashing',
        });
      }

      // ── Security: tempfile without secure methods ──
      if (trimmed.includes('tempfile.mktemp(')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'medium',
          description: 'Race condition: use tempfile.mkstemp() or NamedTemporaryFile instead',
        });
      }

      // ── Bug: division by zero ──
      if ((trimmed.includes('/ len(') || trimmed.includes('/len(')) && !trimmed.includes('if ')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'bug',
          severity: 'high',
          description: 'Potential ZeroDivisionError when collection is empty',
        });
      }

      // ── Bug: bare except ──
      if (/^except\s*:/.test(trimmed)) {
        findings.push({
          line_number: lineNum,
          issue_type: 'bug',
          severity: 'medium',
          description: 'Bare except catches all exceptions including SystemExit and KeyboardInterrupt',
        });
      }

      // ── Bug: mutable default argument ──
      if (/def \w+\(.*=\s*\[\]/.test(trimmed) || /def \w+\(.*=\s*\{\}/.test(trimmed)) {
        findings.push({
          line_number: lineNum,
          issue_type: 'bug',
          severity: 'medium',
          description: 'Mutable default argument — use None and assign inside function',
        });
      }

      // ── Security: base64 decode + exec/eval ──
      if ((trimmed.includes('base64.b64decode') || trimmed.includes('b64decode')) &&
          (trimmed.includes('exec(') || trimmed.includes('eval('))) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'critical',
          description: 'Obfuscated code execution via base64 decode + exec/eval',
        });
      }

      // ── Security: XXE / XML parsing ──
      if (trimmed.includes('xml.etree.ElementTree') || trimmed.includes('ET.parse(') ||
          trimmed.includes('ET.fromstring(')) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'high',
          description: 'Potential XXE vulnerability in XML parsing; use defusedxml',
        });
      }

      // ── Security: SSRF ──
      if ((trimmed.includes('requests.get(') || trimmed.includes('urllib.request.urlopen(')) &&
          !trimmed.includes('https://') && (trimmed.includes('url') || trimmed.includes('URL'))) {
        findings.push({
          line_number: lineNum,
          issue_type: 'security',
          severity: 'high',
          description: 'Potential SSRF: user-controlled URL in outbound request',
        });
      }
    });

    return findings;
  }

  // ══════════════════════════════════════════════════════════════════
  //  FALSE POSITIVE GENERATOR
  // ══════════════════════════════════════════════════════════════════
  const FALSE_POSITIVE_TEMPLATES = [
    { issue_type: 'style', severity: 'low', description: 'Consider adding type hints to function parameters' },
    { issue_type: 'style', severity: 'low', description: 'Line exceeds recommended 79 character limit' },
    { issue_type: 'bug',   severity: 'medium', description: 'Variable may be used before assignment in edge case' },
    { issue_type: 'bug',   severity: 'low', description: 'Return value is not checked for None' },
    { issue_type: 'style', severity: 'low', description: 'Missing docstring for function' },
    { issue_type: 'style', severity: 'low', description: 'Consider using f-string instead of .format()' },
    { issue_type: 'bug',   severity: 'medium', description: 'Potential off-by-one error in loop boundary' },
    { issue_type: 'style', severity: 'low', description: 'Inconsistent indentation style detected' },
  ];

  function generateFalsePositive(lines) {
    const template = FALSE_POSITIVE_TEMPLATES[randInt(0, FALSE_POSITIVE_TEMPLATES.length - 1)];
    const lineNum = randInt(1, lines.length);
    return { line_number: lineNum, ...template };
  }

  // ══════════════════════════════════════════════════════════════════
  //  RANDOMIZED FINDING SELECTION
  // ══════════════════════════════════════════════════════════════════
  function selectFindings(allFindings, settings, lines) {
    const r = settings.randomness; // 0=conservative, 1=balanced, 2=chaotic

    // Keep rate varies by randomness level
    const keepRates = [
      [0.80, 0.90],   // conservative: keep 80-90%
      [0.50, 0.85],   // balanced: keep 50-85%
      [0.30, 0.70],   // chaotic: keep 30-70%
    ];
    const [minKeep, maxKeep] = keepRates[r] || keepRates[1];
    const keepRate = randFloat(minKeep, maxKeep);

    // Randomly select a subset
    let selected = allFindings.filter(() => Math.random() < keepRate);

    // Ensure we always have at least 1 finding (if any were detected)
    if (selected.length === 0 && allFindings.length > 0) {
      selected = [allFindings[randInt(0, allFindings.length - 1)]];
    }

    // Shuffle order
    selected = shuffle(selected);

    // Optionally inject false positives
    if (settings.falsePositives) {
      const fpChance = [0.10, 0.20, 0.35][r] || 0.20;
      if (Math.random() < fpChance) {
        const fp = generateFalsePositive(lines);
        // Insert at random position
        const insertIdx = randInt(0, selected.length);
        selected.splice(insertIdx, 0, fp);
      }
    }

    return selected;
  }

  // ══════════════════════════════════════════════════════════════════
  //  LLM-POWERED FINDING GENERATION
  // ══════════════════════════════════════════════════════════════════
  async function callLLM(codeSnippet, settings) {
    const baseUrl = settings.llmUrl.replace(/\/+$/, '');
    const endpoint = `${baseUrl}/chat/completions`;

    const systemPrompt = `You are a code review expert. Analyze the given Python code and return a JSON array of findings. Each finding must have:
- "line_number": integer (1-indexed)
- "issue_type": one of "style", "bug", "security"
- "severity": one of "low", "medium", "high", "critical"
- "description": brief description of the issue

Return ONLY a valid JSON array, no other text.`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${settings.llmKey}`,
      },
      body: JSON.stringify({
        model: settings.llmModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: `Review this Python code:\n\n\`\`\`python\n${codeSnippet}\n\`\`\`` },
        ],
        temperature: 0.7,
        max_tokens: 2048,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`LLM API error ${response.status}: ${text.substring(0, 200)}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';

    // Parse JSON from response (may be wrapped in markdown code block)
    let jsonStr = content;
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) jsonStr = jsonMatch[1];

    // Try to find a JSON array in the response
    const arrMatch = jsonStr.match(/\[[\s\S]*\]/);
    if (!arrMatch) throw new Error('LLM response did not contain a JSON array');

    const findings = JSON.parse(arrMatch[0]);

    // Validate and sanitize findings
    return findings.filter(f =>
      f.line_number && f.issue_type && f.severity && f.description
    ).map(f => ({
      line_number: parseInt(f.line_number, 10),
      issue_type: ['style', 'bug', 'security'].includes(f.issue_type) ? f.issue_type : 'style',
      severity: ['low', 'medium', 'high', 'critical'].includes(f.severity) ? f.severity : 'medium',
      description: String(f.description).substring(0, 200),
    }));
  }

  function showLLMStatus(msg, isError) {
    const el = llmStatus();
    if (!el) return;
    el.style.display = 'block';
    el.className = 'llm-status ' + (isError ? 'error' : 'connected');
    el.textContent = msg;
  }

  // ══════════════════════════════════════════════════════════════════
  //  SIMULATION RUNNER
  // ══════════════════════════════════════════════════════════════════
  async function runSimulation() {
    if (running) return;
    running = true;
    runBtn().disabled = true;
    resetBtn().disabled = true;

    const settings = readSettingsFromUI();
    saveSettings(settings);

    const taskId = taskSelect().value;
    const diff = taskId.split('_')[0];
    const r = settings.randomness;

    // Reset UI
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
    let allSubmitted = [];

    const codeLines = obs.code_snippet.split('\n');
    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

    // ── Delay ranges by randomness ──
    const delayRange = [
      [500, 700],    // conservative
      [400, 1000],   // balanced
      [300, 1200],   // chaotic
    ][r] || [400, 1000];

    const randomDelay = () => sleep(randInt(delayRange[0], delayRange[1]));

    // ── Decide strategy based on settings + randomness ──
    // Number of submission phases: 2-4 depending on randomness
    const minPhases = [2, 2, 2][r];
    const maxPhases = [2, 3, 4][r];
    const numSubmitPhases = randInt(minPhases, maxPhases);

    // Decide whether to use hints/analysis (randomized based on settings)
    let willUseHint = settings.useHints;
    let willUseAnalysis = settings.useAnalysis;

    // Add randomness to hint/analysis decisions
    if (r >= 1) {
      // Balanced/Chaotic: randomly skip hints sometimes
      if (willUseHint && diff === 'easy' && Math.random() < 0.6) willUseHint = false;
      if (willUseHint && diff === 'medium' && Math.random() < 0.2) willUseHint = false;
      if (willUseAnalysis && diff !== 'hard' && Math.random() < 0.7) willUseAnalysis = false;
    }
    if (r === 0) {
      // Conservative: always use hints for non-easy, always use analysis for hard
      willUseHint = settings.useHints && diff !== 'easy';
      willUseAnalysis = settings.useAnalysis && diff === 'hard';
    }

    // ── Get findings ──
    let allDetectedFindings = [];
    let usedLLM = false;

    if (settings.useLLM && settings.llmUrl && settings.llmKey) {
      // Try LLM mode
      try {
        addLog(0, '*', 'Calling LLM for code analysis...', 0);
        allDetectedFindings = await callLLM(obs.code_snippet, settings);
        usedLLM = true;
        showLLMStatus(`Received ${allDetectedFindings.length} findings from LLM`, false);
      } catch (e) {
        showLLMStatus(`LLM failed: ${e.message} — falling back to heuristics`, true);
        addLog(0, '!', `LLM failed, using heuristic fallback`, 0);
        allDetectedFindings = detectAllFindings(codeLines);
      }
    } else {
      allDetectedFindings = detectAllFindings(codeLines);
    }

    // ── Randomize and split findings across submission phases ──
    let selectedFindings;
    if (usedLLM) {
      // LLM findings: still apply randomization
      selectedFindings = selectFindings(allDetectedFindings, settings, codeLines);
    } else {
      selectedFindings = selectFindings(allDetectedFindings, settings, codeLines);
    }

    // Split findings into phases
    const phases = [];
    if (selectedFindings.length === 0) {
      phases.push([]);
    } else {
      const chunkSize = Math.max(1, Math.ceil(selectedFindings.length / numSubmitPhases));
      for (let i = 0; i < selectedFindings.length; i += chunkSize) {
        phases.push(selectedFindings.slice(i, i + chunkSize));
      }
    }

    // ── Execute simulation phases ──
    let phaseIdx = 0;

    // Phase 1: Submit initial findings
    if (phases[phaseIdx] && phases[phaseIdx].length > 0) {
      await randomDelay();
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const resp = await API.submitFindings(episodeId, phases[phaseIdx], false);
        totalReward = resp.reward || 0;
        const feedbackText = resp.feedback || 'Findings submitted';
        addLog(stepNum, '>', `Submitted ${phases[phaseIdx].length} finding(s) — ${feedbackText}`, resp.reward || 0);
        phases[phaseIdx].forEach(f => highlightLine(f.line_number, true));
        allSubmitted.push(...phases[phaseIdx]);
      } catch (e) {
        addLog(stepNum, 'X', `Step failed: ${e.message}`, 0);
        running = false;
        runBtn().disabled = false;
        resetBtn().disabled = false;
        return;
      }
      phaseIdx++;
    }

    // Optional: Request hint
    if (willUseHint) {
      await randomDelay();
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const hintResp = await API.requestHint(episodeId);
        totalReward = hintResp.reward || totalReward;
        const hintText = hintResp.hint || 'No hint available';
        addLog(stepNum, '?', `Hint: ${hintText}`, -0.05);
      } catch (e) {
        addLog(stepNum, '!', `Hint request failed: ${e.message}`, 0);
      }
    }

    // Optional: Request analysis
    if (willUseAnalysis) {
      await randomDelay();
      stepNum++;
      stepBadge().textContent = `Step ${stepNum}`;
      try {
        const analysisResp = await API.requestAnalysis(episodeId);
        totalReward = analysisResp.reward || totalReward;
        const analysisText = analysisResp.analysis_result || 'No analysis available';
        addLog(stepNum, '@', `Analysis: ${analysisText.substring(0, 120)}...`, -0.10);
      } catch (e) {
        addLog(stepNum, '!', `Analysis request failed: ${e.message}`, 0);
      }
    }

    // Submit remaining phases
    while (phaseIdx < phases.length) {
      const batch = phases[phaseIdx];
      if (batch.length > 0) {
        await randomDelay();
        stepNum++;
        stepBadge().textContent = `Step ${stepNum}`;
        try {
          const resp = await API.submitFindings(episodeId, batch, false);
          totalReward = resp.reward || totalReward;
          addLog(stepNum, '>', `Submitted ${batch.length} more finding(s) — ${resp.feedback || ''}`, resp.reward || 0);
          batch.forEach(f => highlightLine(f.line_number, true));
          allSubmitted.push(...batch);
        } catch (e) {
          addLog(stepNum, '!', `Step failed: ${e.message}`, 0);
        }
      }
      phaseIdx++;
    }

    // Final: Done
    await sleep(randInt(300, 600));
    stepNum++;
    stepBadge().textContent = `Step ${stepNum}`;
    try {
      const doneResp = await API.sendDone(episodeId);
      totalReward = doneResp.reward || totalReward;
      addLog(stepNum, '#', `Episode complete. Final reward: ${totalReward.toFixed(3)}`, totalReward);
    } catch (e) {
      addLog(stepNum, '!', `Done failed: ${e.message}`, 0);
    }

    // ── Grading stats ──
    const totalSubmitted = allSubmitted.length;
    const estimatedGt = diff === 'easy' ? 5 : diff === 'medium' ? 5 : 8;
    const approxMatched = Math.min(totalSubmitted, Math.round(estimatedGt * Math.max(0, totalReward + 0.15)));
    const falsePos = Math.max(0, totalSubmitted - approxMatched);

    updateGrading(approxMatched, estimatedGt, totalSubmitted, falsePos, totalReward);

    running = false;
    runBtn().disabled = false;
    resetBtn().disabled = false;
  }

  // ══════════════════════════════════════════════════════════════════
  //  RESET
  // ══════════════════════════════════════════════════════════════════
  function reset() {
    if (running) return;
    logContainer().innerHTML = '<div class="empty-state" style="padding:24px;"><div class="empty-icon">&#128203;</div>Episode steps will appear here</div>';
    codeViewer().innerHTML = '<div class="empty-state"><div class="empty-icon">&#128196;</div>Select a task and run the simulation</div>';
    gradingPanel().style.display = 'none';
    stepBadge().textContent = 'Step 0';
    taskBadge().textContent = '--';
  }

  // ══════════════════════════════════════════════════════════════════
  //  INITIALIZATION
  // ══════════════════════════════════════════════════════════════════
  function init() {
    // Wire up buttons
    runBtn().addEventListener('click', runSimulation);
    resetBtn().addEventListener('click', reset);

    // Settings toggle
    const toggle = settingsToggle();
    const panel = settingsPanel();
    if (toggle && panel) {
      toggle.addEventListener('click', () => {
        toggle.classList.toggle('open');
        panel.classList.toggle('open');
      });
    }

    // Randomness slider label
    const slider = randomnessSlider();
    if (slider) {
      slider.addEventListener('input', () => {
        randomnessLabel().textContent = RANDOMNESS_LABELS[parseInt(slider.value, 10)] || 'Balanced';
      });
    }

    // LLM toggle visibility
    const llmChk = chkUseLLM();
    if (llmChk) {
      llmChk.addEventListener('change', () => {
        updateLLMVisibility(llmChk.checked);
        updateModeBadge(llmChk.checked);
      });
    }

    // Auto-save settings on any change
    const settingsInputs = [
      randomnessSlider(), chkFalsePos(), chkHints(), chkAnalysis(),
      chkUseLLM(), llmUrl(), llmKey(), llmModel(),
    ];
    settingsInputs.forEach(el => {
      if (!el) return;
      const event = el.type === 'checkbox' || el.type === 'range' ? 'change' : 'blur';
      el.addEventListener(event, () => {
        saveSettings(readSettingsFromUI());
      });
    });

    // Load saved settings
    const saved = loadSettings();
    applySettingsToUI(saved);
  }

  // Auto-init when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { runSimulation, reset };
})();
