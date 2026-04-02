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
  const astSummaryEl  = () => document.getElementById('pgAstSummary');
  const astSummaryGrid= () => document.getElementById('pgAstSummaryGrid');
  const fixToggle     = () => document.getElementById('pgFixToggle');
  const fixArea       = () => document.getElementById('pgFixArea');
  const fixCode       = () => document.getElementById('pgFixCode');
  const submitFixBtn  = () => document.getElementById('pgSubmitFix');

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
  function showInfo(type, text, isHtml = false) {
    const area = infoArea();
    const box = document.createElement('div');
    box.className = `info-box ${type}`;
    if (isHtml) {
      box.innerHTML = text;
    } else {
      box.textContent = text;
    }
    area.appendChild(box);
    // Auto-remove after 30s
    setTimeout(() => box.remove(), 30000);
  }

  // ── Render AST summary panel ──
  function renderAstSummary(summary) {
    const panel = astSummaryEl();
    const grid = astSummaryGrid();
    if (!summary || !panel || !grid) return;

    grid.innerHTML = '';

    const DANGEROUS_MODULES = ['os', 'subprocess', 'pickle', 'eval', 'exec', 'yaml', 'xml'];

    // Functions
    const funcs = summary.functions || [];
    if (funcs.length > 0) {
      const item = _astItem('Functions', `${funcs.length} defined`);
      const tags = document.createElement('div');
      tags.className = 'ast-tags';
      funcs.forEach(f => {
        const tag = document.createElement('span');
        tag.className = 'ast-tag';
        tag.textContent = f;
        tags.appendChild(tag);
      });
      item.appendChild(tags);
      grid.appendChild(item);
    }

    // Imports
    const imports = summary.imports || [];
    if (imports.length > 0) {
      const item = _astItem('Imports', `${imports.length} modules`);
      const tags = document.createElement('div');
      tags.className = 'ast-tags';
      imports.forEach(imp => {
        const tag = document.createElement('span');
        tag.className = 'ast-tag' + (DANGEROUS_MODULES.includes(imp) ? ' dangerous' : '');
        tag.textContent = imp;
        tags.appendChild(tag);
      });
      item.appendChild(tags);
      grid.appendChild(item);
    }

    // Classes
    const classes = summary.classes || [];
    if (classes.length > 0) {
      grid.appendChild(_astItem('Classes', classes.join(', ')));
    }

    // Stats
    grid.appendChild(_astItem('Total Lines', summary.total_lines || '?'));
    grid.appendChild(_astItem('Function Calls', summary.call_count || '?'));

    const dangerCount = summary.dangerous_import_count || 0;
    const dangerItem = _astItem('Dangerous Imports', dangerCount);
    if (dangerCount > 0) {
      dangerItem.querySelector('.ast-summary-item-value').classList.add('danger');
    }
    grid.appendChild(dangerItem);

    panel.style.display = 'block';
  }

  function _astItem(label, value) {
    const item = document.createElement('div');
    item.className = 'ast-summary-item';
    item.innerHTML =
      `<div class="ast-summary-item-label">${label}</div>` +
      `<div class="ast-summary-item-value">${value}</div>`;
    return item;
  }

  // ── Render fix feedback from submit_fix response ──
  function renderFixFeedback(fixFeedback) {
    if (!fixFeedback || !fixFeedback.fixes) return;

    fixFeedback.fixes.forEach(fix => {
      let cssClass, statusLabel;
      if (fix.is_valid) {
        cssClass = 'fix-valid';
        statusLabel = 'VALID';
      } else if (fix.feedback && fix.feedback.toLowerCase().includes('regression')) {
        cssClass = 'fix-regression';
        statusLabel = 'REGRESSION';
      } else {
        cssClass = 'fix-rejected';
        statusLabel = 'REJECTED';
      }

      const html =
        `<div class="fix-result-header">${statusLabel} &mdash; Line ${fix.line} [${fix.check_id}]</div>` +
        `<div class="fix-result-detail">${fix.feedback || 'No details'}</div>` +
        (fix.score !== undefined ? `<div class="fix-result-detail">Score: ${fix.score.toFixed(2)}</div>` : '');

      showInfo(cssClass, html, true);
    });

    if (fixFeedback.total_bonus !== undefined) {
      const bonusText = fixFeedback.total_bonus >= 0
        ? `+${fixFeedback.total_bonus.toFixed(3)}`
        : fixFeedback.total_bonus.toFixed(3);
      showInfo(
        fixFeedback.total_bonus >= 0 ? 'hint' : 'error',
        `Fix total bonus: ${bonusText}`
      );
    }
  }

  // ── Render AST analysis results ──
  function renderAnalysisResults(analysisText) {
    if (!analysisText) return;

    // Parse structured AST analysis output
    // Format: "FINDINGS:\n  Line X: [CHECK_ID] message (confidence: N%)\n..."
    const lines = analysisText.split('\n');
    let html = '';
    let findingCount = 0;

    for (const line of lines) {
      const match = line.match(/Line\s+(\d+):\s*\[([^\]]+)\]\s*(.*?)(?:\s*\(confidence:\s*(\d+)%\))?$/);
      if (match) {
        findingCount++;
        const [, lineNum, checkId, msg, confidence] = match;
        html +=
          `<div class="ast-finding">` +
          `<span class="ast-finding-line">L${lineNum}</span>` +
          `<span class="ast-finding-check">${checkId}</span>` +
          `<span class="ast-finding-msg">${msg.trim()}</span>` +
          (confidence ? `<span class="ast-finding-confidence">${confidence}%</span>` : '') +
          `</div>`;
      }
    }

    if (findingCount > 0) {
      showInfo('analysis',
        `<div style="margin-bottom:6px; font-weight:700;">AST Analysis: ${findingCount} finding(s)</div>` + html,
        true
      );
    } else {
      showInfo('analysis', `AST Analysis: ${analysisText}`);
    }
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

      // Display AST summary if available
      if (obs.ast_summary) {
        renderAstSummary(obs.ast_summary);
      } else {
        astSummaryEl().style.display = 'none';
      }

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
      const resp = await API.runAstAnalysis(episodeId);
      stepNumber = resp.step_number || stepNumber + 1;
      totalReward = resp.reward || totalReward;
      analysisUsed = true;
      updateStats();

      if (resp.analysis_result) {
        renderAnalysisResults(resp.analysis_result);
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

  // ── Submit a code fix via API ──
  async function submitFix() {
    if (!active || selectedLine === null) return;

    const code = fixCode().value.trim();
    if (!code) {
      showInfo('error', 'Fix code is empty. Write corrected code in the textarea.');
      return;
    }

    const finding = {
      line_number: selectedLine,
      issue_type: issueType().value,
      severity: issueSeverity().value,
      description: issueDesc().value || `Fix for line ${selectedLine}`,
      fix_code: code,
    };

    issueForm().style.display = 'none';

    submitFixBtn().disabled = true;
    try {
      const resp = await API.submitFixes(episodeId, [finding]);
      stepNumber = resp.step_number || stepNumber + 1;
      totalReward = resp.reward || totalReward;
      updateStats();

      // Render fix feedback
      if (resp.fix_feedback) {
        renderFixFeedback(resp.fix_feedback);
      }

      // Show general feedback
      if (resp.feedback) {
        showInfo('hint', `Feedback: ${resp.feedback}`);
      }

      if (resp.done) endEpisode(resp);
    } catch (e) {
      showInfo('error', `Fix submission failed: ${e.message}`);
    }

    submitFixBtn().disabled = false;
    selectedLine = null;
    // Reset fix area
    fixCode().value = '';
    fixArea().classList.remove('visible');
    fixToggle().textContent = '+ Attach fix code (optional)';
    submitFixBtn().style.display = 'none';
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
      // Reset fix area
      fixCode().value = '';
      fixArea().classList.remove('visible');
      fixToggle().textContent = '+ Attach fix code (optional)';
      submitFixBtn().style.display = 'none';
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

    // Fix code toggle
    fixToggle().addEventListener('click', () => {
      const area = fixArea();
      const isVisible = area.classList.contains('visible');
      area.classList.toggle('visible', !isVisible);
      fixToggle().textContent = isVisible
        ? '+ Attach fix code (optional)'
        : '- Hide fix code';
      submitFixBtn().style.display = isVisible ? 'none' : 'inline-flex';
    });

    // Submit fix button
    submitFixBtn().addEventListener('click', submitFix);

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
