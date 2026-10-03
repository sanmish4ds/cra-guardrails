#!/usr/bin/env node

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    sessions: path.join(projectRoot, "data/human_cra_transfer/sessions.jsonl"),
    results: path.join(projectRoot, "experiments/results/human_transfer_eval.json"),
    out: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const option = argv[i];
    if (!["--sessions", "--results", "--out"].includes(option)) {
      throw new Error(`Unknown option: ${option}`);
    }
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${option}`);
    }
    args[option.slice(2)] = path.resolve(value);
    i += 1;
  }

  return args;
}

function readJsonl(filePath) {
  return readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line, index) => {
      let row;
      try {
        row = JSON.parse(line);
      } catch (error) {
        throw new Error(`${filePath}:${index + 1}: invalid JSON`, { cause: error });
      }
      if (row.label !== 0 && row.label !== 1) {
        throw new Error(`${filePath}:${index + 1}: label must be 0 or 1`);
      }
      return row;
    });
}

function summarizeSources(sessions) {
  if (sessions.length === 0) {
    throw new Error("The sessions file contains no records.");
  }

  const counts = new Map();
  let missingSourceCount = 0;
  for (const row of sessions) {
    const source =
      typeof row.source === "string" && row.source.trim() !== ""
        ? row.source.trim()
        : "unknown";
    if (source === "unknown") missingSourceCount += 1;
    const sourceCounts = counts.get(source) ?? [0, 0];
    sourceCounts[row.label] += 1;
    counts.set(source, sourceCounts);
  }

  const sourceLabelCounts = Object.fromEntries(
    [...counts.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([source, [negative, positive]]) => [
        source,
        { negative, positive, total: negative + positive },
      ]),
  );
  const mixedSources = [...counts.values()].filter(
    ([negative, positive]) => negative > 0 && positive > 0,
  ).length;
  const sourcesPerLabel = [0, 1].map(
    (label) =>
      [...counts.values()].filter((sourceCounts) => sourceCounts[label] > 0).length,
  );
  const sourceDeterministicallyIdentifiesLabel =
    [...counts.values()].every(
      ([negative, positive]) => negative === 0 || positive === 0,
    ) && sourcesPerLabel.every((count) => count > 0);

  return {
    n_sessions: sessions.length,
    n_sources: counts.size,
    n_sources_with_both_labels: mixedSources,
    sources_per_label: { negative: sourcesPerLabel[0], positive: sourcesPerLabel[1] },
    missing_source_count: missingSourceCount,
    source_label_counts: sourceLabelCounts,
    source_deterministically_identifies_label: sourceDeterministicallyIdentifiesLabel,
    descriptive_in_sample_source_only_auc_not_generalizable:
      sourceDeterministicallyIdentifiesLabel ? 1 : null,
    within_source_class_comparison_identifiable: mixedSources > 0,
    leave_one_source_out_auc_identifiable: mixedSources >= 2,
  };
}

function summarizeTransferResults(resultsPath) {
  if (!resultsPath) return null;
  const results = JSON.parse(readFileSync(resultsPath, "utf8"));
  const selected = {};
  for (const name of ["CRA-Net (lambda=0.05)", "CRA-Net DA"]) {
    const method = results.methods?.[name];
    if (!method) continue;
    selected[name] = Object.fromEntries(
      [
        "auroc",
        "sfpr_at_tpr90",
        "synth_val_threshold",
        "tpr_at_synth_threshold",
        "fpr_at_synth_threshold",
      ]
        .filter((key) => method[key] !== undefined)
        .map((key) => [key, method[key]]),
    );
  }
  return {
    protocol: results.protocol ?? null,
    description: results.description ?? null,
    n_human_total: results.n_human_total ?? null,
    n_human_pos_cosafe: results.n_human_pos_cosafe ?? null,
    n_human_neg_sharegpt: results.n_human_neg_sharegpt ?? null,
    methods: selected,
  };
}

function summarizeReportedSourceCounts(results) {
  const positive = results?.n_human_pos_cosafe;
  const negative = results?.n_human_neg_sharegpt;
  if (
    !Number.isInteger(positive) ||
    positive < 0 ||
    !Number.isInteger(negative) ||
    negative < 0 ||
    positive + negative === 0
  ) {
    throw new Error(
      "Cannot audit the missing sessions file: transfer results must report non-negative integer CoSafe/ShareGPT counts.",
    );
  }

  return {
    n_sessions: positive + negative,
    n_sources: 2,
    n_sources_with_both_labels: 0,
    sources_per_label: {
      negative: Number(negative > 0),
      positive: Number(positive > 0),
    },
    missing_source_count: 0,
    source_label_counts: {
      cosafe: { negative: 0, positive, total: positive },
      sharegpt: { negative, positive: 0, total: negative },
    },
    source_deterministically_identifies_label: positive > 0 && negative > 0,
    descriptive_in_sample_source_only_auc_not_generalizable:
      positive > 0 && negative > 0 ? 1 : null,
    within_source_class_comparison_identifiable: false,
    leave_one_source_out_auc_identifiable: false,
    basis: "aggregate_counts_in_existing_transfer_result",
  };
}

export function auditTransfer(sessions, results = null) {
  const sourceAudit = sessions
    ? summarizeSources(sessions)
    : summarizeReportedSourceCounts(results);
  return {
    audit: "source_label_confounding_v1",
    source_audit: sourceAudit,
    existing_transfer_results: results,
    interpretation:
      "A source-only score computed from the same data is descriptive, not a generalization estimate. If source deterministically identifies the label, pooled transfer metrics cannot distinguish risk detection from source recognition.",
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const results = summarizeTransferResults(args.results);
  const sessions = existsSync(args.sessions) ? readJsonl(args.sessions) : null;
  const report = auditTransfer(sessions, results);
  const output = `${JSON.stringify(report, null, 2)}\n`;

  if (args.out) {
    writeFileSync(args.out, output, { flag: "w" });
    console.error(`Wrote source-confounding audit to ${args.out}`);
  }
  process.stdout.write(output);
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  try {
    main();
  } catch (error) {
    console.error(`[transfer-audit] ${error.message}`);
    process.exitCode = 1;
  }
}
