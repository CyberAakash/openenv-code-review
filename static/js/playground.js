/**
 * playground.js — Interactive code review game.
 * The user plays as the RL agent, clicking lines to report issues.
 */

const Playground = (() => {
  // ── DOM refs ──
  const taskSelect    = () => document.getElementById('pgTaskSelect');
  const startBtn      = () => document.getElementById('pgStartBtn');
  const doneBtn       = () => document.getElementById('pgDoneBtn');
  const codeViewer    = () => document.getElementById('pgCodeViewer');
  const taskBadge     = () => document.getElementById('pgTaskBadge');
  const stepBadge     = () => document.getElementById('pgStepBadge');
  const rewardVal     = () => document.getElementById('pgRewardVal');
  const findingsVal   = () => document.getElementById('pgFindingsVal');
  const rewardBar     = () => document.getElementById('pgRewardBar');
  const rewardLabel   = () => document.getElementById('pgRewardLabel');
  const hintBtn       = () => document.getElementById('pgHintBtn');
  const analysisBtn   = () => document.getElementById('pgAnalysisBtn');
  const issueForm     = () => document.getElementById('pgIssueForm');
  const issueLineBadge= () => document.getElementById('pgIssueLineBadge');
  const issueType     = () => document.getElementById('pgIssueType');
  const issueSeverity = () => document.getElementById('pgIssueSeverity');
  const issueDesc     = () => document.getElementById('pgIssueDesc');
  const submitIssue   = () => document.getElementById('pgSubmitIssue');
  const cancelIssue   = () => document.getElementById('pgCancelIssue');
  const findingsList  = () => document.getElementById('pgFindingsList');
  const findingsCount = () => document.getElementById('pgFindingsCountBadge');
  const infoArea      = () => document.getElementById('pgInfoArea');
  const scoreModal    = () => document.getElementById('pgScoreModal');
  const finalScore    = () => document.getElementById('pgFinalScore');
  const finalFindings = () => document.getElementById('pgFinalFindings');
  const finalSteps    = () => document.getElementById('pgFinalSteps');
  const scoreClose    = () => document.getElementById('pgScoreClose');

  // ── State ──
  let episodeId = null;
  let stepNumber = 0;
  let maxSteps = 10;
  let totalReward = 0;
  let selectedLine = null;
  let findings = [];
  let active = false;
  let hintsUsed = 0;
  let analysisUsed = false;

  // ── Render code with clickable lines ──
  function renderCode(snippet) {
    const viewer = codeViewer();
    viewer.innerHTML = '';
    const lines = snippet.split('\n');
    lines.forEach((text, idx) => {
      const lineEl = document.createElement('div');
      lineEl.className = 'code-line clickable';
      lineEl.dataset.line = idx + 1;

      const marker = document.createElement('span');
      marker.className = 'code-line-marker';

      const numEl = document.createElement('span');
      numEl.className = 'code-line-number';
      numEl.textContent = idx + 1;

      const contentEl = document.createElement('span');
      contentEl.className = 'code-line-content';
      contentEl.textContent = text;

      lineEl.appendChild(marker);
      lineEl.appendChild(numEl);
      lineEl.appendChild(contentEl);

      lineEl.addEventListener('click', () => {
        if (!active) return;
        selectLine(idx + 1);
      });

      viewer.appendChild(lineEl);
    });
  }

  // ── Select a line to report issue ──
  function selectLine(lineNum) {
    // Deselect previous
    const viewer = codeViewer();
    viewer.querySelectorAll('.code-line.highlighted').forEach(el => {
      if (!el.classList.contains('flagged-correct') && !el.classList.contains('flagged-wrong')) {
        el.classList.remove('highlighted');
      }
    });

    // Highlight selected
    const lineEl = viewer.querySelector(`[data-line="${lineNum}"]`);
    if (lineEl) lineEl.classList.add('highlighted');

    selectedLine = lineNum;
    issueLineBadge().textContent = `Line ${lineNum}`;
    issueDesc().value = '';
    issueForm().style.display = 'block';
    issueDesc().focus();
  }

  // ── Update UI stats ──
  function updateStats() {
    stepBadge().textContent = `Step ${stepNumber} / ${maxSteps}`;
    rewardVal().textContent = totalReward.toFixed(3);
    findingsVal().textContent = findings.length;
    findingsCount().textContent = findings.length;

    const pct = Math.max(0, Math.min(100, totalReward * 100));
    const bar = rewardBar();
    bar.style.width = pct + '%';
    bar.classList.toggle('negative', totalReward < 0);
    rewardLabel().textContent = totalReward.toFixed(3);
  }

  // ── Render findings sidebar ──
  function renderFindings() {
    const list = findingsList();
    if (findings.length === 0) {
      list.innerHTML = '<div class="empty-state" style="padding:24px;"><div class="empty-icon">&#128270;</div>Click a code line to add findings</div>';
      return;
    }
    list.innerHTML = '';
    findings.forEach((f, idx) => {
      const item = document.createElement('div');
      item.className = 'finding-item';

      const lineEl = document.createElement('span');
      lineEl.className = 'finding-line';
      lineEl.textContent = `L${f.line_number}`;

      const typeEl = document.createElement('span');
      typeEl.className = `finding-type ${f.issue_type}`;
      typeEl.textContent = f.issue_type;

      const sevEl = document.createElement('span');
      sevEl.className = 'finding-severity';
      sevEl.textContent = f.severity;

      const descEl = document.createElement('span');
      descEl.className = 'finding-desc';
      descEl.textContent = f.description;

      const removeEl = document.createElement('span');
      removeEl.className = 'finding-remove';
      removeEl.textContent = 'x';
      removeEl.title = 'Remove finding';
      removeEl.addEventListener('click', () => removeFinding(idx));

      item.appendChild(lineEl);
      item.appendChild(typeEl);
      item.appendChild(sevEl);
      item.appendChild(descEl);
      item.appendChild(removeEl);
      list.appendChild(item);
    });
  }

  // ── Remove a finding ──
  function removeFinding(idx) {
    findings.splice(idx, 1);
    renderFindings();
    updateStats();
  }

  // ── Show info box ──
  function showInfo(type, text) {
    const area = infoArea();
    const box = document.createElement('div');
    box.className = `info-box ${type}`;
    box.textContent = text;
    area.appendChild(box);
    // Auto-remove after 30s
    setTimeout(() => box.remove(), 30000);
  }

  // ── Start episode ──
  async function startEpisode() {
    const taskId = taskSelect().value;

    startBtn().disabled = true;
    startBtn().textContent = 'Starting...';

    try {
      const obs = await API.reset(taskId);
      episodeId = obs.episode_id;
      stepNumber = obs.step_number || 0;
      maxSteps = obs.max_steps || 10;
      totalReward = 0;
      findings = [];
      selectedLine = null;
      active = true;
      hintsUsed = 0;
      analysisUsed = false;

      renderCode(obs.code_snippet);
      renderFindings();
      updateStats();
      taskBadge().textContent = taskId;

      doneBtn().disabled = false;
      hintBtn().disabled = false;
      analysisBtn().disabled = false;
      issueForm().style.display = 'none';
      infoArea().innerHTML = '';

      startBtn().textContent = 'Restart';
      startBtn().disabled = false;
    } catch (e) {
      showInfo('error', `Failed to start: ${e.message}`);
      startBtn().textContent = 'Start Review';
      startBtn().disabled = false;
    }
  }

  // ── Submit a single finding via API ──
  async function submitFinding() {
    if (!active || selectedLine === null) return;

    const finding = {
      line_number: selectedLine,
      issue_type: issueType().value,
      severity: issueSeverity().value,
      description: issueDesc().value || `Issue on line ${selectedLine}`,
    };

    findings.push(finding);
    renderFindings();
    issueForm().style.display = 'none';

    // Submit to API
    submitIssue().disabled = true;
    try {
      const resp = await API.submitFindings(episodeId, [finding], false);
      stepNumber = resp.step_number || stepNumber + 1;
      totalReward = resp.reward || totalReward;

      // Mark the line
      const lineEl = codeViewer().querySelector(`[data-line="${selectedLine}"]`);
      if (lineEl) {
        lineEl.classList.remove('highlighted');
        lineEl.classList.add('flagged-correct');
        const marker = lineEl.querySelector('.code-line-marker');
        if (marker) marker.textContent = '+';
      }

      // Show feedback
      if (resp.feedback) {
        showInfo('hint', `Feedback: ${resp.feedback}`);
      }

      updateStats();

      // Check if episode ended
      if (resp.done) {
        endEpisode(resp);
      }
    } catch (e) {
      showInfo('error', `Submit failed: ${e.message}`);
    }

    submitIssue().disabled = false;
    selectedLine = null;
  }

  // ── Request hint ──
  async function requestHint() {
    if (!active) return;
    if (hintsUsed >= 3) {
      showInfo('error', 'Maximum hints (3) already used');
      return;
    }

    hintBtn().disabled = true;
    try {
      const resp = await API.requestHint(episodeId);
      stepNumber = resp.step_number || stepNumber + 1;
      totalReward = resp.reward || totalReward;
      hintsUsed++;
      updateStats();

      if (resp.hint) {
        showInfo('hint', `Hint: ${resp.hint}`);
      }
      if (resp.done) endEpisode(resp);
    } catch (e) {
      showInfo('error', `Hint failed: ${e.message}`);
    }

    hintBtn().disabled = hintsUsed >= 3;
  }

  // ── Request analysis ──
  async function requestAnalysis() {
    if (!active || analysisUsed) return;

    analysisBtn().disabled = true;
    try {
      const resp = await API.requestAnalysis(episodeId);
      stepNumber = resp.step_number || stepNumber + 1;
      totalReward = resp.reward || totalReward;
      analysisUsed = true;
      updateStats();

      if (resp.analysis_result) {
        showInfo('analysis', `Analysis: ${resp.analysis_result}`);
      }
      if (resp.done) endEpisode(resp);
    } catch (e) {
      showInfo('error', `Analysis failed: ${e.message}`);
    }
    // Analysis can only be used once
    analysisBtn().disabled = true;
  }

  // ── Finalize episode ──
  async function finalize() {
    if (!active) return;

    doneBtn().disabled = true;
    try {
      const resp = await API.sendDone(episodeId);
      totalReward = resp.reward || totalReward;
      stepNumber = resp.step_number || stepNumber + 1;
      updateStats();
      endEpisode(resp);
    } catch (e) {
      showInfo('error', `Finalize failed: ${e.message}`);
      doneBtn().disabled = false;
    }
  }

  // ── End episode — show score modal ──
  function endEpisode(resp) {
    active = false;
    doneBtn().disabled = true;
    hintBtn().disabled = true;
    analysisBtn().disabled = true;

    // Remove clickable class from all lines
    codeViewer().querySelectorAll('.code-line').forEach(el => {
      el.classList.remove('clickable');
    });

    totalReward = resp.reward || totalReward;

    finalScore().textContent = totalReward.toFixed(3);
    finalFindings().textContent = findings.length;
    finalSteps().textContent = stepNumber;

    scoreModal().classList.add('active');
    startBtn().textContent = 'Start Review';
  }

  // ── Init event listeners ──
  function init() {
    startBtn().addEventListener('click', startEpisode);
    doneBtn().addEventListener('click', finalize);
    submitIssue().addEventListener('click', submitFinding);
    cancelIssue().addEventListener('click', () => {
      issueForm().style.display = 'none';
      selectedLine = null;
      // Remove highlight
      codeViewer().querySelectorAll('.code-line.highlighted').forEach(el => {
        if (!el.classList.contains('flagged-correct') && !el.classList.contains('flagged-wrong')) {
          el.classList.remove('highlighted');
        }
      });
    });
    hintBtn().addEventListener('click', requestHint);
    analysisBtn().addEventListener('click', requestAnalysis);
    scoreClose().addEventListener('click', () => {
      scoreModal().classList.remove('active');
    });

    // Submit on Enter in description
    issueDesc().addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitFinding();
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { startEpisode, finalize };
})();
