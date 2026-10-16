'use strict';

/**
 * Minimal Node.js (18+) client for the guardrail service. Call it from your
 * server, never from the browser: the API key must stay secret, and `subject`
 * must be an identity your server controls (user id), not a client session id.
 */
class GuardrailClient {
  constructor({ baseUrl, apiKey, timeoutMs = 10000, failOpen = false }) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
    this.failOpen = failOpen; // decision to use if the service itself is unreachable
  }

  async post(path, body) {
    try {
      const r = await fetch(this.baseUrl + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': this.apiKey },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (r.status === 401) {
        return { decision: 'block', reasons: ['guardrail service rejected the API key (401); check configuration'] };
      }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      return { decision: this.failOpen ? 'allow' : 'block', reasons: [`guardrail service unavailable: ${e.message}`] };
    }
  }

  checkInput(subject, text) { return this.post('/v1/check/input', { subject, text }); }

  checkOutput(subject, prompt, response) { return this.post('/v1/check/output', { subject, prompt, response }); }

  checkSql(subject, sql) { return this.post('/v1/check/sql', { subject, sql }); }

  reportSqlResult(subject, sql, rowsReturned) {
    return this.post('/v1/report/sql-result', { subject, sql, rows_returned: rowsReturned });
  }
}

module.exports = { GuardrailClient };
