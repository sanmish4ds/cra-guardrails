import test from "node:test";
import assert from "node:assert/strict";
import { auditTransfer } from "../audit_transfer_confounding.mjs";

test("flags labels that are perfectly determined by source", () => {
  const report = auditTransfer([
    { source: "source-a", label: 1 },
    { source: "source-a", label: 1 },
    { source: "source-b", label: 0 },
  ]);

  assert.equal(report.source_audit.source_deterministically_identifies_label, true);
  assert.equal(
    report.source_audit.descriptive_in_sample_source_only_auc_not_generalizable,
    1,
  );
  assert.equal(report.source_audit.within_source_class_comparison_identifiable, false);
  assert.equal(report.source_audit.leave_one_source_out_auc_identifiable, false);
});

test("requires at least two mixed sources for leave-one-source-out AUC", () => {
  const report = auditTransfer([
    { source: "source-a", label: 0 },
    { source: "source-a", label: 1 },
    { source: "source-b", label: 0 },
    { source: "source-b", label: 1 },
  ]);

  assert.equal(report.source_audit.source_deterministically_identifies_label, false);
  assert.equal(report.source_audit.within_source_class_comparison_identifiable, true);
  assert.equal(report.source_audit.leave_one_source_out_auc_identifiable, true);
});

test("counts missing source metadata explicitly", () => {
  const report = auditTransfer([{ label: 0 }, { source: "source-a", label: 1 }]);

  assert.equal(report.source_audit.missing_source_count, 1);
  assert.equal(report.source_audit.source_label_counts.unknown.negative, 1);
});

test("audits reported source counts when the restricted corpus is absent", () => {
  const report = auditTransfer(null, {
    n_human_pos_cosafe: 750,
    n_human_neg_sharegpt: 222,
  });

  assert.equal(report.source_audit.n_sessions, 972);
  assert.equal(report.source_audit.source_deterministically_identifies_label, true);
  assert.equal(
    report.source_audit.basis,
    "aggregate_counts_in_existing_transfer_result",
  );
});

test("does not infer source counts when the restricted corpus and counts are absent", () => {
  assert.throws(
    () => auditTransfer(null, {}),
    /Cannot audit the missing sessions file/,
  );
});
