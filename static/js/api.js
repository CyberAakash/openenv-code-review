/**
 * api.js — Shared API client for the Code Review environment.
 * All calls go to the same origin (the FastAPI server).
 */

const API = (() => {
  const BASE = window.location.origin;

  async function _fetch(path, options = {}) {
    const url = `${BASE}${path}`;
    const resp = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  return {
    /** Health check */
    async health() {
      return _fetch('/health');
    },

    /** Get schema */
    async schema() {
      return _fetch('/schema');
    },

    /** Reset environment for a given task */
    async reset(taskId) {
      return _fetch('/reset', {
        method: 'POST',
        body: JSON.stringify({ task_id: taskId }),
      });
    },

    /** Submit a step action */
    async step(action) {
      return _fetch('/step', {
        method: 'POST',
        body: JSON.stringify({ action }),
      });
    },

    /** Convenience: submit review findings */
    async submitFindings(episodeId, findings, done = false) {
      return this.step({
        action_type: 'review',
        findings,
        done,
        metadata: { episode_id: episodeId },
      });
    },

    /** Convenience: request a hint */
    async requestHint(episodeId) {
      return this.step({
        action_type: 'request_hint',
        findings: [],
        done: false,
        metadata: { episode_id: episodeId },
      });
    },

    /** Convenience: request static analysis */
    async requestAnalysis(episodeId) {
      return this.step({
        action_type: 'request_analysis',
        findings: [],
        done: false,
        metadata: { episode_id: episodeId },
      });
    },

    /** Convenience: send done signal */
    async sendDone(episodeId) {
      return this.step({
        action_type: 'review',
        findings: [],
        done: true,
        metadata: { episode_id: episodeId },
      });
    },
  };
})();
