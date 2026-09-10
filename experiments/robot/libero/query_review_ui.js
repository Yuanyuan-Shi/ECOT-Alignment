(() => {
  "use strict";
  const D = window.D;
  if (!window.localStorage || !D?.review) return;
  const pct = value => value == null ? "—" : `${(100 * value).toFixed(1)}%`;
  const esc = value => String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);

  // Proposal v2 separates task-relative action correctness from faithfulness.
  // Use a new key so approvals of old proposals cannot appear as confirmations
  // of the corrected proposals.
  const STORAGE_KEY = "ecotQueryTaxonomyReviewsV2";
  const TASK_0_3_9_BACKUP_KEY = "ecotTask0Task3Task9LabelsBackupV48";
  const SUBTASK_CHECKPOINT_RULE_VERSION = 3;
  const CONFIRMED = new Set(["approved", "corrected"]);
  const CASES = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5", "Case 6", "Unresolved"];
  const CASE_MAP = {
    "Correct|Correct|Yes": "Case 1",
    "Incorrect|Correct|No": "Case 2",
    "Correct|Incorrect|No": "Case 3",
    "Incorrect|Incorrect|Yes": "Case 4",
    "Incorrect|Incorrect|No": "Case 5",
    "Correct|Alternative valid|No": "Case 6",
  };
  const ACTION_CASE_MAP = {
    "T|T|T": "Case 1",
    "F|T|F": "Case 2",
    "T|F|F": "Case 3",
    "F|F|T": "Case 4",
    "F|F|F": "Case 5",
    "T|T|F": "Case 6",
  };
  const COLORS = {
    "Case 1": "case-1", "Case 2": "case-2", "Case 3": "case-3", "Case 4": "case-4",
    "Case 5": "case-5", "Case 6": "case-6", Ambiguous: "case-ambiguous",
    Unverifiable: "case-unverifiable", Unresolved: "case-ambiguous",
  };
  let reviews = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  const task039Reviews = Object.fromEntries(Object.entries(reviews).filter(([id]) => {
    const taskId = D.review.records?.[id]?.task_id;
    return [0, 3, 9].includes(taskId) || /^libero_90-task(?:0|3|9)-/.test(id);
  }));
  if (Object.keys(task039Reviews).length) {
    localStorage.setItem(TASK_0_3_9_BACKUP_KEY, JSON.stringify({
      schema_version: 1,
      source_key: STORAGE_KEY,
      task_ids: [0, 3, 9],
      saved_at: new Date().toISOString(),
      reviews: task039Reviews,
    }));
  }
  let currentId = null;

  const reasoningValues = D.review.taxonomy.reasoning_values;
  const actionValues = D.review.taxonomy.action_values;
  const faithfulnessValues = D.review.taxonomy.faithfulness_values;
  const queueSelect = document.getElementById("qr-queue");
  const batchSelect = document.getElementById("qr-batch");
  const statusSelect = document.getElementById("qr-status");

  function derivedCase(reasoning, action, faithfulness) {
    if ([reasoning, action, faithfulness].includes("Unverifiable")) return "Unverifiable";
    return CASE_MAP[[reasoning, action, faithfulness].join("|")] || "Ambiguous";
  }

  function overallAlignment(values) {
    if (values.includes("F")) return "F";
    if (values.some(value => !value)) return "Pending";
    const evaluable = values.filter(value => value !== "N/A");
    if (!evaluable.length) return "N/A";
    if (evaluable.includes("Partial")) return "Partial";
    return "T";
  }

  function alignmentToFaithfulness(value) {
    return ({T: "Yes", F: "No", Partial: "Partial", "N/A": "Unverifiable"})[value] || "Ambiguous";
  }

  function binaryReasoning(value) {
    return ({T: "T", Correct: "T", F: "F", Incorrect: "F"})[value] || "";
  }

  function binaryAction(value) {
    return ({T: "T", Correct: "T", "Alternative valid": "T", F: "F", Incorrect: "F"})[value] || "";
  }

  function actionAlignmentCase(reasoning, action, alignment) {
    const binaryAlignment = alignment === "T" ? "T" : ["Partial", "F"].includes(alignment) ? "F" : "";
    return ACTION_CASE_MAP[[binaryReasoning(reasoning), binaryAction(action), binaryAlignment].join("|")] || "Unresolved";
  }

  function isConfirmed(review) { return review && CONFIRMED.has(review.status); }
  function humanCase(id) {
    const review = reviews[id];
    if (!isConfirmed(review)) return null;
    return ["T", "F"].includes(review.reasoning) && ["T", "F"].includes(review.action)
      ? review.derived_case || actionAlignmentCase(review.reasoning, review.action, review.kinematic_alignment)
      : derivedCase(review.reasoning, review.action, review.faithfulness);
  }
  function displayCase(id, proposed) { return humanCase(id) || proposed; }
  function outcomeKey(value) { return ["Ambiguous", "Unverifiable"].includes(value) ? "Unresolved" : value; }
  function cssCase(value) { return COLORS[value] || "case-ambiguous"; }
  function persist() { localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews)); }

  function assignments(queue) {
    const ids = queue === "representative" ? D.review.representative_ids : D.review.diagnostic_ids;
    return ids.map(id => D.review.records[id]);
  }

  function populateBatches() {
    const values = [...new Set(assignments(queueSelect.value).map(record => record.batch))].sort((a, b) => a - b);
    const old = Number(batchSelect.value);
    batchSelect.innerHTML = values.map(value => `<option value="${value}">Batch ${value}</option>`).join("");
    if (values.includes(old)) batchSelect.value = old;
  }

  function visibleIds() {
    return assignments(queueSelect.value)
      .filter(record => Number(record.batch) === Number(batchSelect.value))
      .filter(record => {
        const status = reviews[record.id]?.status || "unreviewed";
        return statusSelect.value === "all" || statusSelect.value === status;
      })
      .map(record => record.id);
  }

  function optionList(values, current) {
    return values.map(value => `<option${value === current ? " selected" : ""}>${esc(value)}</option>`).join("");
  }

  function vectorOverlay(record) {
    const trajectory = record.executed_ee_trajectory_pixels || [];
    const expected = record.expected_motion_pixels || [];
    const red = trajectory.map(point => `${Number(point[0]).toFixed(1)},${Number(point[1]).toFixed(1)}`).join(" ");
    const green = expected.map(point => `${Number(point[0]).toFixed(1)},${Number(point[1]).toFixed(1)}`).join(" ");
    return `<svg viewBox="0 0 224 224" aria-label="trajectory overlay">
      ${green ? `<polyline points="${green}" fill="none" stroke="#20c45a" stroke-width="4" marker-end="url(#arrow)"/>` : ""}
      ${red ? `<polyline points="${red}" fill="none" stroke="#ef233c" stroke-width="4"/>` : ""}
      <defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><polygon points="0 0,7 3.5,0 7" fill="#20c45a"/></marker></defs>
    </svg>`;
  }

  function frame(record, name) {
    return `<div class="review-frame"><strong>${name[0].toUpperCase() + name.slice(1)}</strong>
      <img src="${record.frames[name]}" alt="${name} frame for task ${record.task_id}, query ${record.query_index}">
      ${name === "after" ? vectorOverlay(record) : ""}</div>`;
  }

  function sceneSpatialOverlay(record) {
    const evidence = record?.spatial_evidence || {};
    const point = value => Array.isArray(value) && value.length === 2 && value.every(item => Number.isFinite(Number(item))) ? value.map(Number) : null;
    const origin = point(evidence.world_origin_pixel), target = point(evidence.target_before_pixel);
    const before = point(evidence.ee_before_pixel), after = point(evidence.ee_after_pixel);
    const marker = (value, body) => value ? body(value[0], value[1]) : "";
    return `<div class="scene-map"><img loading="lazy" src="${record.frames.before}" alt="Task ${record.task_id} scene with projected world markers">
      <svg viewBox="0 0 224 224" aria-label="World origin, target, and end-effector positions">
        ${marker(origin, (x, y) => `<g class="origin-marker"><path d="M${x - 6},${y} L${x + 6},${y} M${x},${y - 6} L${x},${y + 6}"/><text x="${x + 7}" y="${y - 7}">origin</text></g>`)}
        ${marker(target, (x, y) => `<g class="target-marker"><text x="${x}" y="${y}">★</text><text x="${x + 9}" y="${y - 9}">${esc(evidence.target_name || "target")}</text></g>`)}
        ${marker(before, (x, y) => `<g class="ee-before-marker"><circle cx="${x}" cy="${y}" r="5"/><text x="${x + 7}" y="${y + 13}">EE start</text></g>`)}
        ${marker(after, (x, y) => `<g class="ee-after-marker"><circle cx="${x}" cy="${y}" r="5"/><text x="${x + 7}" y="${y - 8}">EE end</text></g>`)}
      </svg><div class="scene-legend"><span class="legend-origin">＋ world (0,0,0)</span><span class="legend-target">★ target</span><span class="legend-before">● EE start</span><span class="legend-after">● EE end</span></div>
      <small>Query ${record.query_index} projection; fit residual ${evidence.projection_median_residual_px == null ? "unavailable" : `${Number(evidence.projection_median_residual_px).toFixed(2)} px`}.</small></div>`;
  }

  function reasoningFields(raw) {
    const names = ["PLAN", "SUBTASK REASONING", "SUBTASK", "MOVE REASONING", "MOVE", "GRIPPER", "GRIPPER POSITION", "VISIBLE OBJECTS", "OBJECTS"];
    const pattern = new RegExp(`(?:^|\\s)(${names.join("|")}):\\s*`, "gi");
    const matches = [...String(raw || "").matchAll(pattern)];
    const output = {};
    matches.forEach((match, index) => {
      const start = match.index + match[0].length;
      const end = index + 1 < matches.length ? matches[index + 1].index : raw.length;
      output[match[1].toUpperCase()] = raw.slice(start, end).trim();
    });
    return output;
  }

  function compactReasoning(record) {
    const fields = reasoningFields(record.reasoning_raw);
    const components = [
      ["TASK", record.task_instruction],
      ["PLAN", fields.PLAN],
      ["SUBTASK", fields.SUBTASK || record.claims?.subtask],
      ["MOVE", fields.MOVE],
      ["GRIPPER", fields["GRIPPER POSITION"] || fields.GRIPPER || record.claims?.gripper_command],
      ["OBJECTS", fields["VISIBLE OBJECTS"] || fields.OBJECTS],
    ];
    const lines = components.map(([name, value]) => `<span class="claim-line"><span class="claim-key component-${name.toLowerCase()}">${name}</span>${value ? esc(value) : '<span class="component-missing">not stated</span>'}</span>`).join("");
    return `${lines}
      <details><summary>Exact ECoT trace</summary><pre class="reason">${esc(record.reasoning_raw)}</pre></details>`;
  }

  function actualPrimitive(delta) {
    if (!Array.isArray(delta) || delta.length !== 3) return "trajectory unavailable";
    // This is a human-readable actuation summary, separate from evaluator scoring.
    const threshold = 0.003, parts = [];
    if (delta[0] < -threshold) parts.push("forward");
    if (delta[0] > threshold) parts.push("back");
    if (delta[1] < -threshold) parts.push("left");
    if (delta[1] > threshold) parts.push("right");
    if (delta[2] < -threshold) parts.push("down");
    if (delta[2] > threshold) parts.push("up");
    if (!parts.length) parts.push("no net translation");
    return parts.join(" + ");
  }

  function rotationSummary(rotationVector) {
    const theta = Math.hypot(...rotationVector);
    const axis = theta > 0 ? rotationVector.map(value => value / theta) : [0, 0, 0];
    const threshold = 0.3;
    const componentDirections = [
      {axis: "x", name: "roll"},
      {axis: "y", name: "pitch"},
      {axis: "z", name: "yaw"},
    ];
    const directionLabels = rotationVector.flatMap((value, index) => {
      if (Math.abs(value) > threshold) {
        const component = componentDirections[index];
        return [`${value > 0 ? "+" : "−"}${component.axis} ${component.name} ${Math.abs(value).toFixed(3)} rad`];
      }
      return [];
    });
    const componentLabel = directionLabels.length ? directionLabels.join(" + ") : "no commanded rotation";
    return {
      theta,
      axis,
      actionLabel: componentLabel,
      label: directionLabels.length
        ? `${componentLabel} (rotation about world axis; sign follows right-hand rule; 0.3 rad per-component threshold)`
        : `no material commanded rotation (no summed rx/ry/rz component exceeds ±0.300 rad)`,
    };
  }

  function directionName(axis, sign) {
    return ({x: sign < 0 ? "forward" : "back", y: sign < 0 ? "left" : "right", z: sign < 0 ? "down" : "up"})[axis];
  }

  function parsedMoveVector(record) {
    const move = reasoningFields(record.reasoning_raw).MOVE || "";
    const translation = move.replace(/\b(?:rotate|rotating)\s+(?:left|right|up|upward|down|downward|forward|front|back|backward|behind)\b/gi, "");
    const terms = {
      x: [["forward", "front"], ["back", "backward", "behind"]],
      y: [["left"], ["right"]],
      z: [["down", "downward", "below"], ["up", "upward", "above"]],
    };
    return ["x", "y", "z"].map(axis => {
      const signs = terms[axis].flatMap((words, index) => words.some(word => new RegExp(`\\b${word}\\b`, "i").test(translation)) ? [index === 0 ? -1 : 1] : []);
      return new Set(signs).size === 1 ? signs[0] : 0;
    });
  }

  function moveCosine(record) {
    const expected = parsedMoveVector(record);
    const delta = record.local_state_change?.ee_delta;
    if (!Array.isArray(delta) || delta.length !== 3) return {score: null, reason: "measured EE displacement unavailable"};
    const actual = delta.map(Number);
    if (!actual.every(Number.isFinite)) return {score: null, reason: "measured EE displacement unavailable"};
    const expectedNorm = Math.hypot(...expected), actualNorm = Math.hypot(...actual);
    // Cosine similarity is mathematically undefined for a zero vector. For
    // review scoring, a missing directional MOVE is represented as [0,0,0]
    // and assigned cosine 0 by convention, hence Move Alignment F.
    if (!expectedNorm) return {score: 0, dot: 0, expectedNorm, actualNorm, expected, actual, zeroExpected: true};
    if (!actualNorm) return {score: null, reason: "summed measured EE displacement is zero", expected, actual};
    const dot = expected.reduce((sum, value, index) => sum + value * actual[index], 0);
    return {score: dot / (expectedNorm * actualNorm), dot, expectedNorm, actualNorm, expected, actual};
  }

  function revisedKinematicMetric(record) {
    const result = moveCosine(record);
    if (result.score === null) return {css: "metric-na", text: "N/A", detail: result.reason};
    const aligned = result.score >= 0.5;
    const directionVector = values => `[${values.map(value => value > 0 ? `+${value}` : String(value)).join(", ")}]`;
    const actualMillimetres = result.actual.map(value => value * 1000);
    const deltaVector = values => `[${values.map(value => {
      const millimetres = value * 1000;
      return `${millimetres > 0 ? "+" : ""}${millimetres.toFixed(2)}`;
    }).join(", ")}] mm`;
    const signed = value => value > 0 ? `+${value}` : String(value);
    const dotTerms = result.expected.map((value, index) => `${signed(value)}×${actualMillimetres[index].toFixed(2)}`).join(" ");
    const dotMillimetres = result.dot * 1000;
    const actualNormMillimetres = result.actualNorm * 1000;
    const denominator = result.expectedNorm * actualNormMillimetres;
    const computation = result.zeroExpected
      ? "cos convention = 0.000 because MOVE is [0,0,0]"
      : `cos = MOVE·ΔEE / (||MOVE|| ||ΔEE||)\n= ${dotMillimetres.toFixed(2)} / (${result.expectedNorm.toFixed(3)} × ${actualNormMillimetres.toFixed(2)})\n= ${dotMillimetres.toFixed(2)} / ${denominator.toFixed(2)} = ${result.score.toFixed(3)}`;
    return {
      css: aligned ? "metric-pass" : "metric-fail",
      text: aligned ? "Aligned" : "Not aligned",
      detail: `cosine ${result.score.toFixed(3)}\nMOVE ${directionVector(result.expected)}\nΣ 10-step action ΔEE ${deltaVector(result.actual)}\ndot = ${dotTerms} = ${dotMillimetres.toFixed(2)}\n||MOVE|| = ${result.expectedNorm.toFixed(3)}; ||ΔEE|| = ${actualNormMillimetres.toFixed(2)} mm\n${computation}\n≥ 0.5 aligned; < 0.5 not aligned`,
    };
  }

  function revisedTargetMetric(record) {
    const evidence = record.spatial_evidence || {};
    const score = record.alignment?.target_score;
    const labels = record.alignment?.diagnostics?.failure_labels || [];
    if (score === null || score === undefined || !evidence.target_name) {
      return {css: "metric-na", text: "Target N/A", detail: labels.includes("target_unresolved") ? "target unresolved or ambiguous" : "no evaluable target claim"};
    }
    const progress = Number(evidence.progress), beforeDistance = Number(evidence.distance_before), afterDistance = Number(evidence.distance_after);
    const position = value => Array.isArray(value) ? `[${value.map(item => Number(item).toFixed(4)).join(", ")}]` : "unavailable";
    const target = String(evidence.target_name).replaceAll("_", " ");
    const conclusion = progress > 0 ? `moved closer to ${target}` : progress < 0 ? `moved farther from ${target}` : `no distance change to ${target}`;
    const wrongObject = labels.includes("wrong_object_interaction");
    const mismatch = progress < 0 || wrongObject;
    const scale = Number(evidence.progress_scale);
    return {
      css: mismatch ? "metric-fail" : "metric-pass",
      text: conclusion,
      detail: `score ${Number(score).toFixed(2)} = clip(${progress.toFixed(4)} / ${scale.toFixed(3)}, −1, 1)\nTarget: ${position(evidence.target_before)} m (${target})\nEE Before: ${position(evidence.ee_before)} m\nd_before: ||EE Before − Target|| = ${beforeDistance.toFixed(4)} m\nEE After: ${position(evidence.ee_after)} m\nd_after: ||EE After − Target|| = ${afterDistance.toFixed(4)} m\nProgress: ${beforeDistance.toFixed(4)} − ${afterDistance.toFixed(4)} = ${progress >= 0 ? "+" : ""}${progress.toFixed(4)} m${wrongObject ? "\nwrong-object interaction detected" : ""}`,
    };
  }

  function compactAction(record) {
    const change = record.local_state_change || {}, delta = change.ee_delta;
    const millimetres = Array.isArray(delta) ? delta.map(value => Math.round(Number(value) * 1000)) : [];
    const translationLabel = actualPrimitive(delta);
    const axis = millimetres.length === 3 ? `x ${millimetres[0] >= 0 ? "+" : ""}${millimetres[0]} · y ${millimetres[1] >= 0 ? "+" : ""}${millimetres[1]} · z ${millimetres[2] >= 0 ? "+" : ""}${millimetres[2]} mm` : "x/y/z unavailable";
    const gripperStates = Array.isArray(change.gripper_open)
      ? change.gripper_open.map(value => value === true ? "OPEN" : value === false ? "CLOSED" : "?") : ["?"];
    const gripper = gripperStates.length === 2 && gripperStates[0] === gripperStates[1]
      ? gripperStates[0] : gripperStates.join(" → ");
    const trajectory = record.executed_ee_trajectory || [];
    const physicalSteps = trajectory.slice(1).map((position, index) => position.map((value, axis) => (Number(value) - Number(trajectory[index][axis])) * 1000));
    const accumulated = physicalSteps.reduce((sum, step) => sum.map((value, axis) => value + step[axis]), [0, 0, 0]);
    const signed = (value, precision = 2) => `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(precision)}`;
    const executedLength = Math.min(
      Number(record.executed_chunk_length ?? physicalSteps.length),
      Array.isArray(record.decoded_action_chunk) ? record.decoded_action_chunk.length : 0,
    );
    const executedCommands = (record.decoded_action_chunk || []).slice(0, executedLength);
    // LIBERO's OSC_POSE controller maps each normalized rotation-vector
    // component from [-1, 1] to [-0.5, 0.5] radians. These are requested
    // controller increments, not measured achieved EE-orientation changes.
    const radiansPerRawRotationUnit = 0.5;
    const rotationSteps = executedCommands.map(action => [3, 4, 5].map(index => Number(action[index]) * radiansPerRawRotationUnit));
    const rotationSum = rotationSteps.reduce((sum, step) => sum.map((value, axis) => value + step[axis]), [0, 0, 0]);
    const rotation = rotationSummary(rotationSum);
    const gripperRaw = executedCommands.map(action => Number(action[6])).filter(Number.isFinite);
    const gripperExecuted = gripperRaw.map(value => value > 0.5 ? -1 : value < 0.5 ? 1 : 0);
    const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    const meanGripperRaw = mean(gripperRaw), meanGripperExecuted = mean(gripperExecuted);
    const gripperMeanText = meanGripperRaw === null
      ? "unavailable"
      : meanGripperRaw.toFixed(4);
    const gripperActionLabel = meanGripperExecuted === null
      ? "gripper unavailable"
      : meanGripperExecuted <= -0.5
        ? "open gripper"
        : meanGripperExecuted >= 0.5
          ? "close gripper"
          : "mixed gripper commands";
    const accumulationRows = physicalSteps.map((step, index) => {
      const cumulative = physicalSteps.slice(0, index + 1).reduce((sum, item) => sum.map((value, axis) => value + item[axis]), [0, 0, 0]);
      const rotation = rotationSteps[index];
      const cumulativeRotation = rotationSteps.slice(0, index + 1).reduce((sum, item) => sum.map((value, axis) => value + item[axis]), [0, 0, 0]);
      const rawGrip = gripperRaw[index], executedGrip = gripperExecuted[index];
      const rotationText = rotation
        ? `  cmd Δr [${rotation.map(value => signed(value, 4)).join(", ")}] rad  cumulative [${cumulativeRotation.map(value => signed(value, 4)).join(", ")}] rad  θ ${Math.hypot(...cumulativeRotation).toFixed(4)} rad`
        : "  cmd Δr unavailable";
      const gripText = Number.isFinite(rawGrip)
        ? `  grip raw ${rawGrip.toFixed(4)} → ${executedGrip.toFixed(0)} (${executedGrip < 0 ? "OPEN" : executedGrip > 0 ? "CLOSE" : "NEUTRAL"})`
        : "  grip unavailable";
      return `${String(index + 1).padStart(2, "0")}: measured ΔEE [${step.map(signed).join(", ")}] mm  cumulative [${cumulative.map(signed).join(", ")}] mm${rotationText}${gripText}`;
    }).join("\n");
    return `<div class="action-title"><strong>${esc(translationLabel)} <span class="action-divider">|</span> ${esc(rotation.actionLabel)} <span class="action-divider">|</span> ${esc(gripperActionLabel)}</strong></div>
      <div class="action-vector">${axis} <span class="action-divider">|</span> grip ${gripper}</div>
      <div class="action-raw">Σ ${physicalSteps.length} measured translation ΔEE = [${accumulated.map(signed).join(", ")}] mm</div>
      <div class="action-raw">convention: direction threshold ±3 mm</div>
      <div class="action-raw">Σ ${rotationSteps.length} commanded rotation vector Δr = [${rotationSum.map(value => signed(value, 4)).join(", ")}] rad</div>
      <div class="action-raw">convention: per-component direction threshold ±0.3 radians</div>
      <div class="action-raw">mean raw gripper over ${gripperRaw.length} decoded commands = ${gripperMeanText}</div>
      <div class="action-raw">convention: CLOSE 0, OPEN +1</div>
      <details><summary>Frames + physical step accumulation + decoded 10×7 action</summary><div class="compact-evidence">${["before", "intermediate", "after"].map(name => `<img loading="lazy" src="${record.frames[name]}" alt="${name}">`).join("")}</div>
      <h5>Per-action accumulation</h5><p><small>Translation is measured from simulator EE positions and labeled with a 3 mm dead-band. Rotation-vector components retain their physical axis meaning: rx = roll about x, ry = pitch about y, and rz = yaw about z; signs follow the right-hand rule. Each label requires |component| &gt; 0.3 rad. Achieved EE orientation was not logged. Finite 3D rotations do not generally compose by simple vector addition, so this is a review convention for commanded rotation, not measured net orientation.</small></p><pre class="reason">${esc(accumulationRows || "Physical trajectory unavailable")}</pre>
      <h5>Decoded controller commands</h5>
      <pre class="reason">${esc(JSON.stringify(record.decoded_action_chunk, null, 1))}</pre></details>`;
  }

  function alignmentMetric(record, level) {
    if (level === "kinematic") return revisedKinematicMetric(record);
    if (level === "target") return revisedTargetMetric(record);
    const score = record.alignment?.[`${level}_score`];
    const labels = record.alignment?.diagnostics?.failure_labels || [];
    if (score === null || score === undefined) return {css: "metric-na", text: "N/A", detail: "not evaluable"};
    let mismatch = level === "target" ? score < 0 || labels.includes("wrong_object_interaction") : score < 1;
    const relevant = labels.filter(label => level === "target" ? ["target_moved_away", "target_unresolved", "wrong_object_interaction"].includes(label) : ["subtask_not_completed", "unsupported_subtask"].includes(label));
    return {css: mismatch ? "metric-fail" : "metric-pass", text: mismatch ? "Mismatch" : "Aligned", detail: `score ${Number(score).toFixed(2)}${relevant.length ? ` · ${relevant.join(", ")}` : ""}`};
  }

  function normalizedSubtask(value) {
    return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
  }

  function planSteps(record) {
    const plan = reasoningFields(record.reasoning_raw).PLAN || "";
    return [...plan.matchAll(/(?:^|,\s*)(\d+)\.\s*(.*?)(?=,\s*\d+\.|$)/g)].map(match => ({
      number: Number(match[1]),
      text: match[2].trim(),
    }));
  }

  function finalCompletionUnits(period, episode) {
    if (!episode.success) return 1;
    const finalRecord = period.records[period.records.length - 1];
    const steps = planSteps(finalRecord);
    const current = normalizedSubtask(period.label);
    const index = steps.findIndex(step => normalizedSubtask(step.text) === current);
    // Successful terminal queries receive credit for the explicitly stated
    // current subtask plus every remaining numbered plan step.
    if (index >= 0) return steps.length - index;
    return 1;
  }

  function buildSubtaskReview(records, episode) {
    const periods = [];
    records.forEach(record => {
      const label = record.claims?.subtask || "No stated subtask";
      const key = normalizedSubtask(label);
      const last = periods[periods.length - 1];
      if (!last || last.key !== key) periods.push({key, label, records: [record]});
      else last.records.push(record);
    });
    const rewards = new Map(), checkpoints = new Map();
    periods.forEach((period, index) => {
      const nextPeriod = periods[index + 1];
      const checkpointRecord = nextPeriod ? nextPeriod.records[0] : period.records[period.records.length - 1];
      const kind = nextPeriod ? "transition" : "final";
      const suggestedUnits = kind === "final" ? finalCompletionUnits(period, episode) : 1;
      const subtaskReturnsLater = periods.slice(index + 1).some(later => later.key === period.key);
      const suggestedJudgment = subtaskReturnsLater || (kind === "final" && !episode.success) ? "incomplete" : "complete";
      const storedCheckpoint = reviews[checkpointRecord.id]?.subtask_checkpoint;
      const saved = storedCheckpoint?.rule_version === SUBTASK_CHECKPOINT_RULE_VERSION ? storedCheckpoint : null;
      const judgment = ["complete", "incomplete"].includes(saved?.judgment) ? saved.judgment : suggestedJudgment;
      const completedUnits = judgment === "complete" ? Math.max(1, Number(saved?.completed_units) || suggestedUnits) : 0;
      const checkpointScore = judgment === "complete" ? completedUnits : -1;
      const distributedScore = kind === "final" && judgment === "complete" ? 1 : checkpointScore;
      const perQueryReward = distributedScore / period.records.length;
      const terminalBonus = kind === "final" && judgment === "complete" ? completedUnits - 1 : 0;
      const checkpoint = {
        checkpointRecord,
        completedUnits,
        judgment,
        kind,
        nextLabel: nextPeriod?.label || null,
        perQueryReward,
        period,
        reviewed: Boolean(saved),
        subtaskReturnsLater,
        score: checkpointScore,
        sceneRecord: period.records[period.records.length - 1],
        previousPeriodRecord: kind === "final" && episode.success && completedUnits > 1 && period.records.length > 1 ? period.records[period.records.length - 2] : null,
        terminalBonus,
      };
      period.records.forEach(record => rewards.set(record.id, checkpoint));
      const displayRecord = period.records[period.records.length - 1];
      (checkpoints.get(displayRecord.id) || checkpoints.set(displayRecord.id, []).get(displayRecord.id)).push(checkpoint);
    });
    return {rewards, checkpoints};
  }

  function subtaskCheckpoint(checkpoint) {
    const record = checkpoint.checkpointRecord;
    const sceneRecord = checkpoint.sceneRecord;
    const fields = reasoningFields(record.reasoning_raw);
    const sceneName = "after";
    const proposalReason = checkpoint.subtaskReturnsLater ? "this subtask returns later → this earlier period is proposed incomplete" : "review the scene and reasoning";
    const status = checkpoint.reviewed ? "human-reviewed" : `proposal — ${proposalReason}`;
    const previousScene = checkpoint.previousPeriodRecord ? `<div class="previous-period-scene"><strong>Previous query ${checkpoint.previousPeriodRecord.query_index} after-scene</strong><img loading="lazy" src="${checkpoint.previousPeriodRecord.frames.after}" alt="Simulator scene after task ${checkpoint.previousPeriodRecord.task_id}, query ${checkpoint.previousPeriodRecord.query_index}"></div>` : "";
    const finalSceneLabel = checkpoint.previousPeriodRecord ? `<strong>Final query ${sceneRecord.query_index} after-scene</strong>` : "";
    const scoreExplanation = checkpoint.terminalBonus > 0
      ? `Checkpoint score +${checkpoint.score}: current subtask +1 ÷ ${checkpoint.period.records.length} = +${checkpoint.perQueryReward.toFixed(3)} per period query; ${checkpoint.terminalBonus} additional remaining ${checkpoint.terminalBonus === 1 ? "subtask" : "subtasks"} = +${checkpoint.terminalBonus} on final query ${sceneRecord.query_index}.`
      : `Checkpoint score ${checkpoint.score > 0 ? "+" : ""}${checkpoint.score}; distributed over ${checkpoint.period.records.length} previous-subtask ${checkpoint.period.records.length === 1 ? "query" : "queries"} = ${checkpoint.perQueryReward > 0 ? "+" : ""}${checkpoint.perQueryReward.toFixed(3)} each.`;
    return `<details class="subtask-checkpoint" open><summary>${checkpoint.kind === "transition" ? `At query ${record.query_index}, label changes to “${esc(checkpoint.nextLabel)}”` : "Final episode checkpoint"}</summary>
      ${previousScene}
      ${finalSceneLabel}
      <img loading="lazy" src="${sceneRecord.frames[sceneName]}" alt="Simulator scene after task ${sceneRecord.task_id}, query ${sceneRecord.query_index}">
      <span><strong>Previous subtask:</strong> ${esc(checkpoint.period.label)}</span>
      <span><strong>Original ECoT subtask reasoning:</strong> ${esc(fields["SUBTASK REASONING"] || "not stated")}</span>
      <label>Previous subtask<select data-subtask-judgment="${esc(record.id)}"><option value="complete"${checkpoint.judgment === "complete" ? " selected" : ""}>Completed (+)</option><option value="incomplete"${checkpoint.judgment === "incomplete" ? " selected" : ""}>Not completed (−1)</option></select></label>
      <label>Completed subtasks<input data-subtask-units="${esc(record.id)}" type="number" min="0" max="20" step="1" value="${checkpoint.completedUnits}"${checkpoint.judgment === "incomplete" ? " disabled" : ""}></label>
      <span><strong>${scoreExplanation}</strong></span>
      <small>${status} · scene is after query ${sceneRecord.query_index}; transition reasoning is from query ${record.query_index}</small></details>`;
  }

  function subtaskCell(record, reviewModel) {
    const reward = reviewModel.rewards.get(record.id);
    const checkpointMarkup = (reviewModel.checkpoints.get(record.id) || []).map(subtaskCheckpoint).join("");
    const isFinalRecord = reward.kind === "final" && record.id === reward.sceneRecord.id;
    const queryReward = reward.perQueryReward + (isFinalRecord ? reward.terminalBonus : 0);
    const css = queryReward > 0 ? "metric-pass" : "metric-fail";
    const sign = queryReward > 0 ? "+" : "";
    const bonusDetail = isFinalRecord && reward.terminalBonus ? `\nterminal remaining-subtask bonus +${reward.terminalBonus}` : "";
    return `<td class="align-cell ${css}"><strong>${sign}${queryReward.toFixed(3)} reward</strong>
      <span class="metric-detail">${esc(reward.period.label)}\nperiod queries ${reward.period.records[0].query_index}–${reward.period.records[reward.period.records.length - 1].query_index}\nperiod base ${reward.perQueryReward > 0 ? "+" : ""}${reward.perQueryReward.toFixed(3)}${bonusDetail}\ncheckpoint total ${reward.score > 0 ? "+" : ""}${reward.score}</span>${checkpointMarkup}</td>`;
  }

  function compactOptions(kind, current) {
    const choices = [["", "—"], ["T", "T"], ["F", "F"]];
    return choices.map(([value, label]) => `<option value="${value}"${value === current ? " selected" : ""}>${label}</option>`).join("");
  }

  function componentAlignmentOptions(current, allowPartial = false) {
    const choices = [["", "—"], ["T", "T"], ...(allowPartial ? [["Partial", "Partial"]] : []), ["F", "F"], ["N/A", "N/A"]];
    return choices.map(([value, label]) => `<option value="${value}"${value === current ? " selected" : ""}>${label}</option>`).join("");
  }

  function componentAlignmentValue(value, allowPartial = false) {
    return ["T", "F", "N/A", ...(allowPartial ? ["Partial"] : [])].includes(value) ? value : "";
  }

  function initialSuggestion(record, reviewModel) {
    const initial = record.initial_labels || {};
    const reward = reviewModel.rewards.get(record.id);
    const isFinalRecord = reward?.kind === "final" && record.id === reward.sceneRecord.id;
    const subtaskReward = reward ? reward.perQueryReward + (isFinalRecord ? reward.terminalBonus : 0) : null;
    const spatial = record.spatial_evidence || {};
    const rawProgress = spatial.progress;
    const progress = Number(rawProgress);
    const hasProgress = rawProgress !== null && rawProgress !== undefined && Number.isFinite(progress);

    const moveResult = moveCosine(record);
    const score = moveResult.score;
    const moveAlignment = score === null ? "N/A" : score >= 0.5 ? "T" : "F";
    const targetScore = record.alignment?.target_score;
    const wrongObject = (record.alignment?.diagnostics?.failure_labels || []).includes("wrong_object_interaction");
    const targetAlignment = componentAlignmentValue(initial.target_alignment)
      || (targetScore === null || targetScore === undefined ? "N/A" : Number(targetScore) > 0 && !wrongObject ? "T" : "F");
    const subtaskAlignment = subtaskReward === null ? "N/A" : subtaskReward > 0 ? "T" : "F";
    const targetEvidence = hasProgress ? `${progress >= 0 ? "+" : "−"}${Math.abs(progress * 1000).toFixed(1)} mm` : "unresolved";
    const rewardEvidence = subtaskReward === null ? "unresolved" : `${subtaskReward >= 0 ? "+" : "−"}${Math.abs(subtaskReward).toFixed(3)} reward`;
    return {
      kinematic_alignment: moveAlignment,
      target_alignment: targetAlignment,
      subtask_alignment: subtaskAlignment,
      explanation: `Move ${moveAlignment} (cos ${score === null ? "N/A" : score.toFixed(3)}). Target ${targetAlignment} (${targetEvidence}). Subtask ${subtaskAlignment} (${rewardEvidence}).`,
    };
  }

  function compactHumanCell(record, reviewModel) {
    const review = reviews[record.id] || {};
    const suggested = initialSuggestion(record, reviewModel);
    const savedMoveAlignment = componentAlignmentValue(review.kinematic_alignment);
    const componentValues = [
      savedMoveAlignment || suggested.kinematic_alignment,
      componentAlignmentValue(review.subtask_alignment || suggested.subtask_alignment),
    ];
    return `<div class="domain-labels move-labels">
      <label>Move Alignment<select data-field="kinematic_alignment">${componentAlignmentOptions(componentValues[0])}</select></label>
      </div><div class="domain-labels subtask-labels">
      <label>Subtask Alignment<select data-field="subtask_alignment">${componentAlignmentOptions(componentValues[1])}</select></label>
      </div>
      <button data-compact-save="corrected">Save labels</button> <button data-compact-save="uncertain">Uncertain</button>
      <div class="initial-rationale"><strong>Initial suggestion:</strong> ${esc(suggested.explanation)}</div>
      <div class="compact-status">${esc(review.status || "unreviewed")}</div>`;
  }

  function renderCompactEpisodes() {
    const root = document.getElementById("qr-compact-episodes");
    if (!root) return;
    const selected = new Set(D.review.compact_episode_ids || []);
    const episodes = D.review.episode_timelines.filter(episode => selected.has(episode.episode_id));
    root.innerHTML = episodes.map((episode, episodeIndex) => {
      const records = episode.queries.map(query => D.review.records[query.id]).filter(Boolean);
      const initial = records[0];
      const subtaskReview = buildSubtaskReview(records, episode);
      const rows = records.map(record => {
        const metrics = ["kinematic"].map(level => alignmentMetric(record, level));
        return `<tr data-compact-id="${esc(record.id)}"><th>${record.query_index}</th><td class="reason-cell">${compactReasoning(record)}</td>
          <td class="action-cell">${compactAction(record)}</td>${metrics.map(metric => `<td class="align-cell ${metric.css}"><strong>${metric.text}</strong><span class="metric-detail">${esc(metric.detail)}</span></td>`).join("")}${subtaskCell(record, subtaskReview)}
          <td class="human-cell">${compactHumanCell(record, subtaskReview)}</td></tr>`;
      }).join("");
      return `<details class="compact-episode"${episodeIndex === 0 ? " open" : ""}><summary>Task ${episode.task_id} · ${episode.success ? "SUCCESS" : "FAILURE"} · ${episode.queries.length} policy queries — ${esc(episode.task_instruction)}</summary>
        <div class="compact-scene">${initial ? sceneSpatialOverlay(initial) : ""}
          <div><strong>Task scene</strong><p>${esc(episode.task_instruction)}</p><p>Task ${episode.task_id} · seed ${episode.seed} · episode ${episode.success ? "SUCCESS" : "FAILURE"}</p><p><small>Fixed third-person RGB captured immediately before policy query 0.</small></p></div></div>
        <div class="table-wrap"><table class="step-table"><thead><tr><th>Step</th><th>ECoT Reasoning (6 components, generated for that query)</th><th>Actual action (Σ executed chunk)</th><th>Move Alignment</th><th>Subtask</th><th>Human labels</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
    }).join("");
    root.querySelectorAll("[data-subtask-judgment], [data-subtask-units]").forEach(input => input.onchange = () => {
      const id = input.dataset.subtaskJudgment || input.dataset.subtaskUnits;
      const judgment = root.querySelector(`[data-subtask-judgment="${CSS.escape(id)}"]`).value;
      const unitsInput = root.querySelector(`[data-subtask-units="${CSS.escape(id)}"]`);
      const completedUnits = judgment === "complete" ? Math.max(1, Number(unitsInput.value) || 1) : 0;
      reviews[id] = {...(reviews[id] || {}), subtask_checkpoint: {rule_version: SUBTASK_CHECKPOINT_RULE_VERSION, judgment, completed_units: completedUnits, updated_at: new Date().toISOString()}};
      persist();
      renderCompactEpisodes();
      renderProgress();
    });
    root.querySelectorAll("button[data-compact-save]").forEach(button => button.onclick = () => saveCompactRow(button.closest("tr"), button.dataset.compactSave));
  }

  function saveCompactRow(row, status) {
    const id = row.dataset.compactId, record = D.review.records[id], proposal = record.proposal;
    const values = Object.fromEntries([...row.querySelectorAll("select[data-field]")].map(item => [item.dataset.field, item.value]));
    values.faithfulness = alignmentToFaithfulness(values.kinematic_alignment);
    reviews[id] = {...(reviews[id] || {}), status, ...values, confidence: proposal.confidence, evidence: proposal.evidence, decisive_error: false, updated_at: new Date().toISOString()};
    persist();
    row.querySelector(".compact-status").textContent = status;
    renderProgress();
  }

  function updateDerived() {
    const result = derivedCase(
      document.getElementById("qr-reasoning").value,
      document.getElementById("qr-action").value,
      document.getElementById("qr-faithfulness").value,
    );
    const output = document.getElementById("qr-derived");
    output.textContent = result;
    output.className = `tag ${cssCase(result)}`;
  }

  function renderCard(id) {
    const record = D.review.records[id];
    if (!record) {
      document.getElementById("qr-card").innerHTML = "<p>No queries match this filter.</p>";
      currentId = null;
      return;
    }
    currentId = id;
    const proposal = record.proposal;
    const review = reviews[id] || {};
    const reasoning = review.reasoning || proposal.reasoning;
    const action = review.action || proposal.action;
    const faithfulness = review.faithfulness || proposal.faithfulness;
    const confidence = review.confidence || proposal.confidence;
    const evidence = review.evidence ?? proposal.evidence;
    const labels = record.alignment.diagnostics?.failure_labels || [];
    document.getElementById("qr-card").innerHTML = `<article class="card">
      <h3>${esc(record.queue)} queue · batch ${record.batch} · item ${record.position}</h3>
      <p><strong>Task ${record.task_id}, query ${record.query_index}</strong> · seed ${record.seed} · episode ${record.episode_success ? "success" : "failure"}<br>${esc(record.task_instruction)}</p>
      <div class="review-frames">${frame(record, "before")}${frame(record, "intermediate")}${frame(record, "after")}</div>
      <p><span class="tag">green = stated motion</span><span class="tag bad">red = executed end-effector trajectory</span></p>
      <details open><summary><strong>Exact policy reasoning</strong></summary><pre class="reason">${esc(record.reasoning_raw)}</pre></details>
      <details><summary><strong>Parsed claims, action tokens, and local state change</strong></summary>
        <h4>Parsed claims</h4><pre class="reason">${esc(JSON.stringify(record.claims, null, 2))}</pre>
        <h4>Action evidence</h4><pre class="reason">${esc(JSON.stringify({action_tokens: record.action_tokens, decoded_action_chunk: record.decoded_action_chunk, executed_ee_trajectory: record.executed_ee_trajectory}, null, 2))}</pre>
        <h4>Local state change</h4><pre class="reason">${esc(JSON.stringify(record.local_state_change, null, 2))}</pre>
      </details>
      <div class="proposal"><strong>Agent proposal: ${proposal.derived_case} · ${proposal.confidence} confidence</strong><br>
        Reasoning=${proposal.reasoning}; Action=${proposal.action}; Faithfulness=${proposal.faithfulness}<br>${esc(proposal.evidence)}<br>
        Diagnostics: ${labels.map(label => `<span class="tag bad">${esc(label)}</span>`).join("") || "none"}</div>
      <h4>Human judgment</h4><div class="judgments">
        <label>Reasoning<select id="qr-reasoning">${optionList(reasoningValues, reasoning)}</select></label>
        <label>Action<select id="qr-action">${optionList(actionValues, action)}</select></label>
        <label>Faithfulness<select id="qr-faithfulness">${optionList(faithfulnessValues, faithfulness)}</select></label>
        <label>Confidence<select id="qr-confidence">${optionList(["High", "Medium", "Low"], confidence)}</select></label>
      </div><p><strong>Automatically derived:</strong> <span id="qr-derived" class="tag"></span></p>
      <label><strong>Evidence for all three judgments</strong><textarea id="qr-evidence">${esc(evidence)}</textarea></label>
      <label><input id="qr-decisive" type="checkbox"${review.decisive_error ? " checked" : ""}> Human-confirmed decisive episode error</label>
      <div class="review-actions"><button id="qr-approve">Approve proposal (A)</button><button id="qr-correct">Save correction (C)</button><button id="qr-uncertain">Mark uncertain (U)</button></div>
      <p>Review state: <strong>${esc(review.status || "unreviewed")}</strong> · ID: ${esc(id)}</p>
    </article>`;
    ["qr-reasoning", "qr-action", "qr-faithfulness"].forEach(name => document.getElementById(name).onchange = updateDerived);
    document.getElementById("qr-approve").onclick = approve;
    document.getElementById("qr-correct").onclick = correct;
    document.getElementById("qr-uncertain").onclick = uncertain;
    updateDerived();
  }

  function formReview(status, useProposal = false) {
    const record = D.review.records[currentId];
    const proposal = record.proposal;
    const result = {
      status,
      reasoning: useProposal ? proposal.reasoning : document.getElementById("qr-reasoning").value,
      action: useProposal ? proposal.action : document.getElementById("qr-action").value,
      faithfulness: useProposal ? proposal.faithfulness : document.getElementById("qr-faithfulness").value,
      confidence: useProposal ? proposal.confidence : document.getElementById("qr-confidence").value,
      evidence: useProposal ? proposal.evidence : document.getElementById("qr-evidence").value.trim(),
      decisive_error: document.getElementById("qr-decisive").checked,
      updated_at: new Date().toISOString(),
    };
    result.derived_case = derivedCase(result.reasoning, result.action, result.faithfulness);
    reviews[currentId] = {...(reviews[currentId] || {}), ...result};
    persist();
    renderAll(false);
  }

  function approve() { if (currentId) formReview("approved", true); }
  function correct() { if (currentId) formReview("corrected"); }
  function uncertain() { if (currentId) formReview("uncertain"); }

  function move(offset) {
    const ids = visibleIds();
    if (!ids.length) return renderCard(null);
    const index = Math.max(0, ids.indexOf(currentId));
    renderCard(ids[(index + offset + ids.length) % ids.length]);
  }

  function renderProgress() {
    const compactIds = new Set();
    const compactEpisodes = new Set(D.review.compact_episode_ids || []);
    D.review.episode_timelines.filter(episode => compactEpisodes.has(episode.episode_id)).forEach(episode => episode.queries.forEach(query => compactIds.add(query.id)));
    const compactReviewed = [...compactIds].filter(id => reviews[id]?.status).length;
    const compactProgress = document.getElementById("qr-compact-progress");
    if (compactProgress) compactProgress.innerHTML = `<strong>${compactReviewed}/${compactIds.size}</strong> steps labeled`;
    if (!queueSelect || !batchSelect) return;
    const queue = assignments(queueSelect.value);
    const batch = queue.filter(record => Number(record.batch) === Number(batchSelect.value));
    const counts = {approved: 0, corrected: 0, uncertain: 0};
    queue.forEach(record => { if (counts[reviews[record.id]?.status] !== undefined) counts[reviews[record.id].status] += 1; });
    const batchDone = batch.filter(record => reviews[record.id]?.status).length;
    const confirmed = counts.approved + counts.corrected;
    const output = document.getElementById("qr-progress");
    if (!output) return;
    output.innerHTML = `<div><strong>${batchDone}/${batch.length}</strong><br>current batch reviewed</div>
      <div><strong>${confirmed}/${queue.length}</strong><br>queue confirmed</div><div><strong>${counts.corrected}</strong><br>corrected labels</div>
      <div><strong>${counts.uncertain}</strong><br>needs adjudication</div>`;
  }

  function renderEpisodeOptions() {
    const select = document.getElementById("qr-episode");
    if (select.options.length) return;
    select.innerHTML = D.review.episode_timelines.map(episode => `<option value="${episode.episode_id}">Task ${episode.task_id} · ${episode.success ? "success" : "failure"}</option>`).join("");
  }

  function renderTimeline() {
    const episode = D.review.episode_timelines.find(item => item.episode_id === document.getElementById("qr-episode").value) || D.review.episode_timelines[0];
    const milestones = episode.milestones;
    const humanDecisive = episode.queries
      .filter(item => reviews[item.id]?.decisive_error && isConfirmed(reviews[item.id]))
      .map(item => item.query).sort((a, b) => a - b)[0];
    const dots = episode.queries.map(item => {
      const shown = displayCase(item.id, item.case);
      const confirmed = humanCase(item.id);
      return `<button class="timeline-dot ${cssCase(shown)}" data-review-id="${item.in_review_queue ? item.id : ""}" title="Query ${item.query}: ${shown}${confirmed ? " (human confirmed)" : " (agent proposed)"}">${item.query}</button>`;
    }).join("");
    document.getElementById("qr-timeline").innerHTML = `<p><strong>Task ${episode.task_id}:</strong> ${esc(episode.task_instruction)} · seed ${episode.seed} · ${episode.success ? "SUCCESS" : "FAILURE"}</p>
      <div class="timeline">${dots}</div><p>Cases present: ${episode.cases_present.join(", ")} · counts: ${esc(JSON.stringify(episode.case_counts))}</p>
      <p>Proposed first non-Case-1: ${milestones.first_non_case1 ?? "none"}; reasoning error: ${milestones.first_reasoning_error ?? "none"}; wrong action: ${milestones.first_wrong_action ?? "none"}; decisive failure: ${milestones.proposed_decisive_failure ?? "none"}; recovery: ${milestones.recovery ?? "none"}.<br>
      First human-confirmed decisive error: ${humanDecisive ?? "pending"}. Policy recovered after a proposed wrong action: ${milestones.recovery != null ? `yes, at query ${milestones.recovery}` : "not detected"}.</p>`;
    document.querySelectorAll("#qr-timeline [data-review-id]").forEach(button => button.onclick = () => {
      const id = button.dataset.reviewId;
      if (!id) return;
      const record = D.review.records[id];
      queueSelect.value = record.queue;
      populateBatches();
      batchSelect.value = record.batch;
      statusSelect.value = "all";
      renderAll(false, id);
    });
  }

  function percentile(values, p) {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * p;
    const lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  }
  function mulberry32(seed) {
    return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function weightedRate(items, outcome) {
    const denominator = items.reduce((sum, item) => sum + item.analysis_weight, 0);
    return denominator ? items.filter(item => outcomeKey(humanCase(item.id)) === outcome).reduce((sum, item) => sum + item.analysis_weight, 0) / denominator : null;
  }
  function bootstrapCI(items, outcome) {
    const groups = items.reduce((result, item) => {
      (result[item.episode_id] ||= []).push(item);
      return result;
    }, {});
    const episodes = Object.keys(groups), rng = mulberry32(20260902), values = [];
    for (let replicate = 0; replicate < 2000; replicate++) {
      const sample = [];
      for (let index = 0; index < episodes.length; index++) sample.push(...groups[episodes[Math.floor(rng() * episodes.length)]]);
      values.push(weightedRate(sample, outcome));
    }
    return [percentile(values, 0.025), percentile(values, 0.975)];
  }

  function renderHumanStats() {
    const representative = assignments("representative");
    const confirmed = representative.filter(record => isConfirmed(reviews[record.id]));
    const output = document.getElementById("qr-human-stats");
    if (!output) return;
    if (!confirmed.length) {
      output.innerHTML = "<p><strong>Pending:</strong> no representative judgments have been human-confirmed. Agent–human agreement and confusion matrix are intentionally not fabricated.</p>";
      return;
    }
    const complete = confirmed.length === representative.length;
    const dimensions = ["reasoning", "action", "faithfulness"];
    const agreement = Object.fromEntries(dimensions.map(name => [name, confirmed.filter(record => reviews[record.id][name] === record.proposal[name]).length]));
    const caseAgreement = confirmed.filter(record => outcomeKey(humanCase(record.id)) === outcomeKey(record.proposal.derived_case)).length;
    let rows = CASES.map(outcome => {
      const matching = confirmed.filter(record => outcomeKey(humanCase(record.id)) === outcome);
      const successful = confirmed.filter(record => record.episode_success);
      const failed = confirmed.filter(record => !record.episode_success);
      const episodes = new Set(confirmed.map(record => record.episode_id));
      const episodesWith = new Set(matching.map(record => record.episode_id));
      const ci = complete ? bootstrapCI(confirmed, outcome) : null;
      return `<tr><th>${outcome}</th><td>${matching.length}/${confirmed.length} (${pct(weightedRate(confirmed, outcome))})</td>
        <td>${matching.filter(record => record.episode_success).length}/${successful.length} (${pct(weightedRate(successful, outcome))})</td>
        <td>${matching.filter(record => !record.episode_success).length}/${failed.length} (${pct(weightedRate(failed, outcome))})</td>
        <td>${episodesWith.size}/${episodes.size} (${pct(episodesWith.size / episodes.size)})</td><td>${ci ? ci.map(pct).join(" to ") : "Pending complete representative review"}</td></tr>`;
    }).join("");
    const humanCases = CASES;
    const confusion = CASES.map(agentCase => `<tr><th>${agentCase}</th>${humanCases.map(human => `<td>${confirmed.filter(record => outcomeKey(record.proposal.derived_case) === agentCase && outcomeKey(humanCase(record.id)) === human).length}</td>`).join("")}</tr>`).join("");
    const examples = CASES.slice(0, 6).map(outcome => {
      const ids = confirmed.filter(record => outcomeKey(humanCase(record.id)) === outcome).slice(0, 3).map(record => `${record.id}`).join("; ");
      return `<li><strong>${outcome}:</strong> ${esc(ids || "pending")}</li>`;
    }).join("");
    output.innerHTML = `<p class="pilot"><strong>${confirmed.length}/${representative.length} representative queries confirmed.</strong> ${complete ? "Probability-weighted human estimates and clustered intervals are complete." : "Partial-review percentages may be selection-biased; do not cite them as prevalence."}</p>
      <p>Agreement — reasoning ${agreement.reasoning}/${confirmed.length} (${pct(agreement.reasoning / confirmed.length)}); action ${agreement.action}/${confirmed.length} (${pct(agreement.action / confirmed.length)}); faithfulness ${agreement.faithfulness}/${confirmed.length} (${pct(agreement.faithfulness / confirmed.length)}); derived case ${caseAgreement}/${confirmed.length} (${pct(caseAgreement / confirmed.length)}). Corrected reviews: ${confirmed.filter(record => reviews[record.id].status === "corrected").length}/${confirmed.length}.</p>
      <div class="table-wrap"><table><thead><tr><th>Human case</th><th>Representative total</th><th>Successful episodes</th><th>Failed episodes</th><th>Episodes containing</th><th>95% CI</th></tr></thead><tbody>${rows}</tbody></table></div>
      <h4>Agent proposal → human-confirmed confusion matrix</h4><div class="table-wrap"><table><thead><tr><th>Agent \\ Human</th>${humanCases.map(value => `<th>${value}</th>`).join("")}</tr></thead><tbody>${confusion}</tbody></table></div>
      <h4>Representative confirmed examples</h4><ul>${examples}</ul>`;
  }

  function renderAll(reset = true, requestedId = null) {
    renderProgress();
    renderTimeline();
    renderHumanStats();
    const ids = visibleIds();
    renderCard(requestedId && ids.includes(requestedId) ? requestedId : (!reset && ids.includes(currentId) ? currentId : ids[0]));
  }

  if (queueSelect) {
    queueSelect.onchange = () => { populateBatches(); statusSelect.value = "all"; renderAll(); };
    batchSelect.onchange = () => renderAll();
    statusSelect.onchange = () => renderAll();
    document.getElementById("qr-prev").onclick = () => move(-1);
    document.getElementById("qr-next").onclick = () => move(1);
    document.getElementById("qr-episode").onchange = renderTimeline;
  }
  document.getElementById("qr-export").onclick = () => {
    const payload = {schema_version: 1, experiment_label: D.review.experiment_label, sampling: D.review.sampling, exported_at: new Date().toISOString(), reviews};
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = "ecot-query-taxonomy-annotations.json"; link.click();
    URL.revokeObjectURL(link.href);
  };
  document.getElementById("qr-import").onchange = event => {
    const file = event.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result);
        if (payload.schema_version !== 1 || typeof payload.reviews !== "object") throw new Error("unsupported annotation schema");
        reviews = {...reviews, ...payload.reviews}; persist(); renderCompactEpisodes(); renderProgress();
        if (queueSelect) renderAll();
      } catch (error) { window.alert(`Could not import annotations: ${error.message}`); }
    };
    reader.readAsText(file);
  };
  if (queueSelect) document.addEventListener("keydown", event => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.key === "j" || event.key === "ArrowRight") move(1);
    else if (event.key === "k" || event.key === "ArrowLeft") move(-1);
    else if (event.key === "a") approve();
    else if (event.key === "c") correct();
    else if (event.key === "u") uncertain();
  });

  renderCompactEpisodes();
  renderProgress();
  if (queueSelect) {
    populateBatches();
    renderEpisodeOptions();
    renderAll();
  }
})();
