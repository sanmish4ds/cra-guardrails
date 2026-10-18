'use strict';

/**
 * Server-side glue between an HTTP API and cra-telemetry.js: records each turn,
 * counts guardrail blocks as refusal turns, and returns non-blocking soft notices.
 *
 * Extracted from SQLclMCP sql-learn-server.js (origin/main, PR #28/#29) so the
 * CRA integration lives with the rest of the CRA code. Reference only; CRA
 * notices are advisory and are not an enforcement layer.
 *
 * Usage:
 *   const { createCraTelemetryFromEnv } = require('./cra-telemetry');
 *   const glue = createCraSessionGlue(createCraTelemetryFromEnv(__dirname));
 *   const craMeta = glue.craClientMetaFromBody(req.body);
 *   res.json({ ...answer, ...glue.craTurnResponseFields(craMeta, turn) });
 */

const CRA_SOFT_WARN_S1 = Number(process.env.CRA_SOFT_WARN_S1 || 0.85);
const CRA_SOFT_WARN_S2 = Number(process.env.CRA_SOFT_WARN_S2 || 0.2);
const CRA_SOFT_WARN_S3 = Number(process.env.CRA_SOFT_WARN_S3 || 0.35);
const CRA_SOFT_WARN_CRA = Number(process.env.CRA_SOFT_WARN_CRA || 0.45);

function craSoftNoticesFromEvent(ev) {
  if (!ev) return [];
  const n = [];
  if (ev.injection_signal) {
    n.push('CRA: this turn matched injection-style wording in the question or tutor reply. Double-check you are not being manipulated.');
  }
  if (typeof ev.S1 === 'number' && ev.S1 >= CRA_SOFT_WARN_S1) {
    n.push('CRA: high intent drift (S1) vs your session anchor—your questions may have moved far from the original goal.');
  }
  if (typeof ev.S2 === 'number' && ev.S2 >= CRA_SOFT_WARN_S2) {
    n.push('CRA: the tutor text is accumulating sensitive-looking patterns (S2). Avoid sharing real secrets; treat examples as synthetic.');
  }
  if (typeof ev.S3 === 'number' && ev.S3 >= CRA_SOFT_WARN_S3) {
    n.push('CRA: refusal / policy language in the tutor is shifting (S3). The model may be tightening or relaxing refusals across turns.');
  }
  if (typeof ev.CRA === 'number' && ev.CRA >= CRA_SOFT_WARN_CRA) {
    n.push('CRA: composite score is elevated—review the answer before relying on it or running SQL on a real database.');
  }
  return n;
}

function craSignalSnapshot(ev) {
  if (!ev) return null;
  return {
    turn: ev.turn,
    S1: ev.S1,
    S2: ev.S2,
    S3: ev.S3,
    CRA: ev.CRA,
    injection_signal: !!ev.injection_signal,
    weights: ev.weights || null,
  };
}

function craClientMetaFromBody(data) {
  const sessionId = data.cra_session_id || data.craSessionId;
  const declaredIntent = data.cra_declared_intent != null ? data.cra_declared_intent : data.craDeclaredIntent;
  const channel = data.cra_channel != null ? data.cra_channel : data.craChannel;
  return {
    sessionId: sessionId ? String(sessionId).trim() : '',
    declaredIntent: declaredIntent != null ? String(declaredIntent) : '',
    channel: channel ? String(channel).trim() : 'api',
  };
}

function createCraSessionGlue(craTelemetry) {
  /**
   * Record one CRA turn (no-op without cra_session_id) and return the response fields
   * { cra_soft_notices, cra_signals } computed for *this* session's turn.
   */
  function craTurnResponseFields(craMeta, turn) {
    if (!craMeta || !craMeta.sessionId) return {};
    const ev = craTelemetry.recordTurn({
      ...turn,
      sessionId: craMeta.sessionId,
      declaredIntent: craMeta.declaredIntent,
      channel: craMeta.channel || turn.channel || 'api',
    });
    const out = {};
    const soft = craSoftNoticesFromEvent(ev);
    if (soft.length) out.cra_soft_notices = soft;
    const snap = craSignalSnapshot(ev);
    if (snap) out.cra_signals = snap;
    return out;
  }

  /**
   * Blocked attempts still count toward session risk: recorded as a refusal turn so S1 sees the
   * question, the injection flag sees its wording, and S3 sees the refusal → compliance gradient.
   */
  function craBlockedTurnFields(craMeta, { question, code, message, mode, channel, eventChannel }) {
    return craTurnResponseFields(craMeta, {
      question,
      tutorResponse: `[guardrails] Request not allowed (${code}): ${message || ''}`,
      sql: '',
      mode,
      source: 'guardrails',
      success: false,
      eventType: 'guardrail_block',
      blocked: code,
      channel: channel || eventChannel,
    });
  }

  return { craTurnResponseFields, craBlockedTurnFields, craClientMetaFromBody };
}

module.exports = {
  createCraSessionGlue,
  craSoftNoticesFromEvent,
  craSignalSnapshot,
  craClientMetaFromBody,
};
