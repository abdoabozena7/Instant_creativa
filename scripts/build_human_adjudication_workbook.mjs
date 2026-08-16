import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(import.meta.dirname, "..");
const evalDir = path.join(projectRoot, "data", "eval");
const outputDir = path.join(projectRoot, "outputs", "ng12_blind_adjudication_v1");
const previewDir = path.join(outputDir, "_previews");
const outputPath = path.join(outputDir, "ng12_human_adjudication_v1.xlsx");

function loadJsonl(text) {
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function citedEvidence(packet, labels) {
  if (!labels.length) return "";
  const wanted = new Set(labels);
  return packet.evidence
    .filter((item) => wanted.has(item.label))
    .map((item) => `${item.label} | ${item.citation}\n${item.text}`)
    .join("\n\n");
}

function applyHeader(range, fill = "#123B36") {
  range.format = {
    fill,
    font: { bold: true, color: "#FFFFFF", size: 10 },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: fill } },
  };
}

function applyTitle(sheet, range, title, subtitle) {
  range.merge();
  range.values = [[title]];
  range.format = {
    fill: "#123B36",
    font: { bold: true, color: "#FFFFFF", size: 20 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
  sheet.getRange("A2:H2").merge();
  sheet.getRange("A2:H2").values = [[subtitle]];
  sheet.getRange("A2:H2").format = {
    fill: "#DCEBE7",
    font: { color: "#294A45", italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:H2").format.rowHeight = 34;
}

function styleInputRange(range) {
  range.format = {
    fill: "#FFF2CC",
    font: { color: "#3F3520" },
    wrapText: true,
    verticalAlignment: "top",
    borders: { insideHorizontal: { style: "thin", color: "#E4D8B1" } },
  };
}

const packets = loadJsonl(
  await fs.readFile(path.join(evalDir, "adjudication_packets_v1.jsonl"), "utf8"),
);
const freeze = JSON.parse(
  await fs.readFile(path.join(evalDir, "evaluation_freeze.json"), "utf8"),
);
const report = JSON.parse(
  await fs.readFile(path.join(evalDir, "blind_e2e_report_v1.json"), "utf8"),
);
const provisionalVerdicts = loadJsonl(
  await fs.readFile(path.join(evalDir, "provisional_claim_adjudication_v1.jsonl"), "utf8"),
);
const verdictByCase = new Map(provisionalVerdicts.map((item) => [item.case_id, item]));

const claimRows = [];
const evidenceRows = [];
for (const packet of packets) {
  const proposedClaims = verdictByCase.get(packet.case_id)?.claims ?? [];
  for (const [claimIndex, claim] of proposedClaims.entries()) {
    const claimId = `C${claimIndex + 1}`;
    const citedLabels = claim.cited_labels ?? [];
    claimRows.push([
      `${packet.case_id}-${claimId}`,
      packet.case_id,
      claimId,
      packet.scope_group,
      packet.expected_behavior,
      packet.question,
      claim.claim,
      citedLabels.join(", "),
      citedEvidence(packet, citedLabels),
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
  }
  for (const evidence of packet.evidence) {
    evidenceRows.push([
      packet.case_id,
      evidence.label,
      evidence.chunk_id,
      evidence.source_version,
      evidence.authority_priority,
      evidence.recommendation_id ?? "",
      evidence.content_type,
      evidence.citation,
      evidence.text,
    ]);
  }
}

const workbook = Workbook.create();
workbook.comments.setSelf({ displayName: "NG12 Reviewer" });

const guide = workbook.worksheets.add("Review Guide");
const summary = workbook.worksheets.add("Metrics Summary");
const cases = workbook.worksheets.add("Case Review");
const claims = workbook.worksheets.add("Claim Review");
const evidence = workbook.worksheets.add("Evidence Index");
guide.showGridLines = false;
applyTitle(
  guide,
  guide.getRange("A1:H1"),
  "NG12 blind evaluation · human adjudication",
  "Evidence adjudication only. This workbook does not provide clinical advice and must not be used for patient-care decisions.",
);
guide.getRange("A4:H4").merge();
guide.getRange("A4:H4").values = [["Review order"]];
applyHeader(guide.getRange("A4:H4"));
guide.getRange("A5:H10").values = [
  ["1", "Review independently", "Reviewer 1 and Reviewer 2 complete their own columns without seeing or copying the other judgment.", "", "", "", "", ""],
  ["2", "Judge behavior", "On Case Review, decide whether the system correctly answered, said evidence was insufficient, or refused as required.", "", "", "", "", ""],
  ["3", "Judge claims", "On Claim Review, mark whether each atomic claim is supported by supplied evidence. Do not use outside clinical knowledge.", "", "", "", "", ""],
  ["4", "Judge citation binding", "For claims with cited labels, decide whether at least one cited passage supports the full claim—not merely the topic.", "", "", "", "", ""],
  ["5", "Adjudicate disagreements", "A third reviewer resolves disagreements and fills only the Final columns. Final values drive the official metrics.", "", "", "", "", ""],
  ["6", "Export and score", "Keep the frozen run unchanged. Export Case Review and Claim Review as CSV only after all required Final fields are complete.", "", "", "", "", ""],
];
guide.getRange("A5:A10").format = { font: { bold: true, color: "#147D70", size: 12 }, horizontalAlignment: "center" };
guide.getRange("B5:B10").format = { font: { bold: true, color: "#17211F" }, verticalAlignment: "top" };
guide.getRange("C5:H10").merge(true);
guide.getRange("A5:H10").format.wrapText = true;
guide.getRange("A5:H10").format.rowHeight = 42;
guide.getRange("A12:H12").merge();
guide.getRange("A12:H12").values = [["Decision rubric"]];
applyHeader(guide.getRange("A12:H12"));
guide.getRange("A13:H18").values = [
  ["Field", "TRUE", "FALSE", "NOT_APPLICABLE", "UNCERTAIN", "Rule", "", ""],
  ["Behavior correct", "Answer/refusal behavior matches the gold requirement", "Wrong behavior or unjustified definitive answer", "Never", "Reviewer cannot resolve from packet", "Judge the response as delivered, not an improved interpretation", "", ""],
  ["Claim supported", "Full atomic meaning follows directly from supplied evidence", "Any material qualifier, exclusion, workflow, or conclusion is added", "Non-claim text only", "Evidence is ambiguous", "Partial support is FALSE; explain the unsupported part", "", ""],
  ["Citation entails", "At least one nearby cited label supports the full claim", "Labels are absent, irrelevant, or only topically related", "No citation was expected for non-clinical scope text", "Binding is ambiguous", "Citation syntax validity is a separate deterministic metric", "", ""],
  ["Current accuracy", "Current 2026 guidance governs action/threshold", "Historical text overrides or distorts current guidance", "No current-vs-history issue", "Version relation is unclear", "2015 evidence may explain but cannot override 2026", "", ""],
  ["Severity", "MINOR: does not alter clinical action", "MAJOR: could change action, threshold, scope, or certainty", "NONE: no error", "", "Use NONE / MINOR / MAJOR", "", ""],
];
applyHeader(guide.getRange("A13:H13"), "#356D65");
guide.getRange("A14:A18").format.font = { bold: true };
guide.getRange("A13:H18").format.wrapText = true;
guide.getRange("A13:H18").format.rowHeight = 50;
guide.getRange("A20:H20").merge();
guide.getRange("A20:H20").values = [[`Frozen architecture: ${freeze.architecture_sha256} · ${packets.length} cases · ${claimRows.length} claims`]];
guide.getRange("A20:H20").format = { fill: "#EEE7D8", font: { color: "#7B5435", bold: true }, wrapText: true };
guide.getRange("A:A").format.columnWidth = 7;
guide.getRange("B:B").format.columnWidth = 22;
guide.getRange("C:H").format.columnWidth = 18;
guide.freezePanes.freezeRows(4);

summary.showGridLines = false;
applyTitle(
  summary,
  summary.getRange("A1:H1"),
  "Official metrics · human review gate",
  "Blank rates mean adjudication is incomplete. Only Final columns are included; provisional same-model triage is excluded.",
);
summary.getRange("A4:E4").values = [["Metric", "Positive", "Reviewed", "Rate", "Target"]];
applyHeader(summary.getRange("A4:E4"));
const caseEnd = packets.length + 5;
const claimEnd = claimRows.length + 5;
summary.getRange("A5:E9").values = [
  ["Answer behavior accuracy", "", "", "", "Maximise"],
  ["Current-guideline accuracy", "", "", "", "100%"],
  ["Claim support rate", "", "", "", "Maximise"],
  ["Unsupported claim rate", "", "", "", "Minimise"],
  ["Citation accuracy", "", "", "", "≥98%"],
];
summary.getRange("B5:C9").formulas = [
  [`=COUNTIF('Case Review'!$I$6:$I$${caseEnd},"TRUE")`, `=COUNTIF('Case Review'!$I$6:$I$${caseEnd},"TRUE")+COUNTIF('Case Review'!$I$6:$I$${caseEnd},"FALSE")`],
  [`=COUNTIF('Case Review'!$L$6:$L$${caseEnd},"TRUE")`, `=COUNTIF('Case Review'!$L$6:$L$${caseEnd},"TRUE")+COUNTIF('Case Review'!$L$6:$L$${caseEnd},"FALSE")`],
  [`=COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"TRUE")`, `=COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"TRUE")+COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"FALSE")`],
  [`=COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"FALSE")`, `=COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"TRUE")+COUNTIF('Claim Review'!$L$6:$L$${claimEnd},"FALSE")`],
  [`=COUNTIF('Claim Review'!$O$6:$O$${claimEnd},"TRUE")`, `=COUNTIF('Claim Review'!$O$6:$O$${claimEnd},"TRUE")+COUNTIF('Claim Review'!$O$6:$O$${claimEnd},"FALSE")`],
];
summary.getRange("D5").formulas = [["=IF(C5=0,\"\",B5/C5)"]];
summary.getRange("D5:D9").fillDown();
summary.getRange("D5:D9").format.numberFormat = "0.0%";
summary.getRange("A11:E11").values = [["Frozen deterministic baseline", "Result", "Target", "Status", "Experiment"]];
applyHeader(summary.getRange("A11:E11"), "#356D65");
const d = report.deterministic_metrics;
summary.getRange("A12:E18").values = [
  ["Scope classification", d.scope_classification_accuracy, 0.99, "Below target", "A"],
  ["Correct refusal", d.correct_refusal_rate, 0.95, "Below target", "A"],
  ["False refusal", d.false_refusal_rate, 0.02, "Within target", "A"],
  ["Citation-label validity", d.citation_label_validity_rate, 0.98, "Below target", "B"],
  ["Retrieval Recall@5", d.retrieval_recall_at_5, 0.95, "Within target", "Frozen"],
  ["Current-guideline accuracy", d.current_guideline_accuracy, 1, "At target", "Frozen"],
  ["End-to-end P50 (seconds)", d.latency_ms.end_to_end_p50 / 1000, "Observe", "Baseline", "Frozen"],
];
summary.getRange("B12:C17").format.numberFormat = "0.0%";
summary.getRange("B18").format.numberFormat = "0.00";
summary.getRange("A20:H20").merge();
summary.getRange("A20:H20").values = [["Decision rule: do not run Experiments A, B, or C until required Final judgments are complete and adjudicated. Run each experiment in isolation against this same frozen baseline."]];
summary.getRange("A20:H20").format = { fill: "#FCE4D6", font: { color: "#8B3D2C", bold: true }, wrapText: true };
summary.getRange("A20:H20").format.rowHeight = 42;
summary.getRange("A:A").format.columnWidth = 31;
summary.getRange("B:D").format.columnWidth = 15;
summary.getRange("E:E").format.columnWidth = 15;
summary.getRange("F:H").format.columnWidth = 12;
summary.freezePanes.freezeRows(4);

cases.showGridLines = false;
cases.getRange("A1:O1").merge();
cases.getRange("A1:O1").values = [["Case-level behavior and source-authority review"]];
applyHeader(cases.getRange("A1:O1"));
cases.getRange("A2:O2").merge();
cases.getRange("A2:O2").values = [["Yellow columns are reviewer inputs. Final columns are completed only after disagreement resolution and drive Metrics Summary."]];
cases.getRange("A2:O2").format = { fill: "#FFF2CC", font: { color: "#6A5624", italic: true }, wrapText: true };
cases.getRange("A5:O5").values = [[
  "Case ID", "Scope group", "Category", "Expected behavior", "Question", "Answer",
  "R1 behavior correct", "R2 behavior correct", "Final behavior correct",
  "R1 current accuracy", "R2 current accuracy", "Final current accuracy",
  "Final claim list complete", "Final failure types", "Adjudicator notes",
]];
applyHeader(cases.getRange("A5:O5"));
cases.getRange(`A6:O${caseEnd}`).values = packets.map((packet) => [
  packet.case_id,
  packet.scope_group,
  packet.category,
  packet.expected_behavior,
  packet.question,
  packet.answer,
  "", "", "", "", "", "", "", "", "",
]);
for (const col of ["G", "H", "I", "J", "K", "L", "M", "N", "O"]) {
  styleInputRange(cases.getRange(`${col}6:${col}${caseEnd}`));
}
for (const col of ["G", "H", "I"]) {
  cases.getRange(`${col}6:${col}${caseEnd}`).dataValidation = {
    rule: { type: "list", values: ["TRUE", "FALSE", "UNCERTAIN"] },
  };
}
for (const col of ["J", "K", "L"]) {
  cases.getRange(`${col}6:${col}${caseEnd}`).dataValidation = {
    rule: { type: "list", values: ["TRUE", "FALSE", "NOT_APPLICABLE", "UNCERTAIN"] },
  };
}
cases.getRange(`M6:M${caseEnd}`).dataValidation = {
  rule: { type: "list", values: ["TRUE", "FALSE", "UNCERTAIN"] },
};
cases.getRange(`A6:O${caseEnd}`).format.verticalAlignment = "top";
cases.getRange(`E6:F${caseEnd}`).format.wrapText = true;
cases.getRange(`A6:O${caseEnd}`).format.rowHeight = 58;
cases.getRange("A:A").format.columnWidth = 12;
cases.getRange("B:D").format.columnWidth = 17;
cases.getRange("E:E").format.columnWidth = 42;
cases.getRange("F:F").format.columnWidth = 58;
cases.getRange("G:L").format.columnWidth = 18;
cases.getRange("M:M").format.columnWidth = 20;
cases.getRange("N:N").format.columnWidth = 25;
cases.getRange("O:O").format.columnWidth = 38;
cases.freezePanes.freezeRows(5);
cases.freezePanes.freezeColumns(1);

claims.showGridLines = false;
claims.getRange("A1:Q1").merge();
claims.getRange("A1:Q1").values = [["Claim-level support and citation-binding review"]];
applyHeader(claims.getRange("A1:Q1"));
claims.getRange("A2:Q2").merge();
claims.getRange("A2:Q2").values = [["The 134 claim rows are model-proposed decomposition only; model verdicts are excluded. Reviewers must verify completeness on Case Review. Support and citation entailment are separate judgments."]];
claims.getRange("A2:Q2").format = { fill: "#FFF2CC", font: { color: "#6A5624", italic: true }, wrapText: true };
claims.getRange("A5:Q5").values = [[
  "Claim key", "Case ID", "Claim ID", "Scope group", "Expected behavior", "Question", "Atomic claim",
  "Cited labels", "Cited evidence", "R1 supported", "R2 supported", "Final supported",
  "R1 citation entails", "R2 citation entails", "Final citation entails", "Final severity", "Reviewer notes",
]];
applyHeader(claims.getRange("A5:Q5"));
claims.getRange(`A6:Q${claimEnd}`).values = claimRows;
for (const col of ["J", "K", "L", "M", "N", "O", "P", "Q"]) {
  styleInputRange(claims.getRange(`${col}6:${col}${claimEnd}`));
}
for (const col of ["J", "K", "L"]) {
  claims.getRange(`${col}6:${col}${claimEnd}`).dataValidation = {
    rule: { type: "list", values: ["TRUE", "FALSE", "NOT_APPLICABLE", "UNCERTAIN"] },
  };
}
for (const col of ["M", "N", "O"]) {
  claims.getRange(`${col}6:${col}${claimEnd}`).dataValidation = {
    rule: { type: "list", values: ["TRUE", "FALSE", "NOT_APPLICABLE", "UNCERTAIN"] },
  };
}
claims.getRange(`P6:P${claimEnd}`).dataValidation = {
  rule: { type: "list", values: ["NONE", "MINOR", "MAJOR", "UNCERTAIN"] },
};
claims.getRange(`A6:Q${claimEnd}`).format.verticalAlignment = "top";
claims.getRange(`F6:I${claimEnd}`).format.wrapText = true;
claims.getRange(`A6:Q${claimEnd}`).format.rowHeight = 72;
claims.getRange("A:A").format.columnWidth = 17;
claims.getRange("B:E").format.columnWidth = 15;
claims.getRange("F:F").format.columnWidth = 38;
claims.getRange("G:G").format.columnWidth = 42;
claims.getRange("H:H").format.columnWidth = 14;
claims.getRange("I:I").format.columnWidth = 72;
claims.getRange("J:O").format.columnWidth = 18;
claims.getRange("P:P").format.columnWidth = 15;
claims.getRange("Q:Q").format.columnWidth = 38;
claims.freezePanes.freezeRows(5);
claims.freezePanes.freezeColumns(2);

evidence.showGridLines = false;
evidence.getRange("A1:I1").merge();
evidence.getRange("A1:I1").values = [["Retrieved evidence supplied to the frozen generator"]];
applyHeader(evidence.getRange("A1:I1"));
evidence.getRange("A3:I3").values = [[
  "Case ID", "Label", "Chunk ID", "Source version", "Authority", "Recommendation ID", "Content type", "Citation", "Source text",
]];
applyHeader(evidence.getRange("A3:I3"));
const evidenceEnd = evidenceRows.length + 3;
evidence.getRange(`A4:I${evidenceEnd}`).values = evidenceRows;
evidence.getRange(`A4:I${evidenceEnd}`).format.verticalAlignment = "top";
evidence.getRange(`H4:I${evidenceEnd}`).format.wrapText = true;
evidence.getRange(`A4:I${evidenceEnd}`).format.rowHeight = 54;
evidence.getRange("A:B").format.columnWidth = 12;
evidence.getRange("C:C").format.columnWidth = 35;
evidence.getRange("D:G").format.columnWidth = 18;
evidence.getRange("H:H").format.columnWidth = 45;
evidence.getRange("I:I").format.columnWidth = 78;
evidence.freezePanes.freezeRows(3);
evidence.freezePanes.freezeColumns(2);

const summaryInspection = await workbook.inspect({
  kind: "table",
  sheetId: "Metrics Summary",
  range: "A1:E20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 5,
  maxChars: 5000,
});
console.log(summaryInspection.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 3000,
});
console.log(formulaErrors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Review Guide", "Metrics Summary", "Case Review", "Claim Review", "Evidence Index"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.75, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.toLowerCase().replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const artifact = await SpreadsheetFile.exportXlsx(workbook);
await fs.mkdir(outputDir, { recursive: true });
await artifact.save(outputPath);
console.log(JSON.stringify({ outputPath, cases: packets.length, claims: claimRows.length, evidenceRows: evidenceRows.length }));
