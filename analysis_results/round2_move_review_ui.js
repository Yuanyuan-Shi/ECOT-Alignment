(() => {
  "use strict";
  const root = document.getElementById("round2-move-review");
  const configNode = document.getElementById("round2-config");
  if (!root || !configNode || !window.localStorage) return;
  const config = JSON.parse(configNode.textContent);
  const key = config.storage_key;
  let reviews = JSON.parse(localStorage.getItem(key) || "{}");
  const rows = [...root.querySelectorAll("[data-round2-row]")];

  function persist() { localStorage.setItem(key, JSON.stringify(reviews)); }
  function applyReviews() {
    rows.forEach(row => {
      const id = row.dataset.id, review = reviews[id];
      const select = row.querySelector("[data-round2-label]");
      select.value = review?.label || select.dataset.default;
      row.querySelector(".compact-status").textContent = review?.status || "unreviewed";
    });
  }
  function progress() {
    const counts = {reviewed: 0, uncertain: 0};
    Object.values(reviews).forEach(review => { if (counts[review.status] !== undefined) counts[review.status] += 1; });
    document.getElementById("round2-progress").textContent = `${counts.reviewed} reviewed · ${counts.uncertain} uncertain · ${rows.length} total`;
  }
  function filters() {
    const task = document.getElementById("round2-task").value;
    const outcome = document.getElementById("round2-outcome").value;
    const status = document.getElementById("round2-status").value;
    root.querySelectorAll("[data-round2-episode]").forEach(episode => {
      const taskMatch = !task || episode.dataset.task === task;
      const outcomeMatch = !outcome || episode.dataset.outcome === outcome;
      const episodeRows = [...episode.querySelectorAll("[data-round2-row]")];
      episodeRows.forEach(row => {
        const rowStatus = reviews[row.dataset.id]?.status || "unreviewed";
        row.style.display = !status || rowStatus === status ? "" : "none";
      });
      const hasVisibleRow = episodeRows.some(row => row.style.display !== "none");
      episode.style.display = taskMatch && outcomeMatch && hasVisibleRow ? "" : "none";
    });
  }
  root.querySelectorAll("[data-round2-save]").forEach(button => button.onclick = () => {
    const row = button.closest("[data-round2-row]");
    const select = row.querySelector("[data-round2-label]");
    reviews[row.dataset.id] = {label: select.value, status: button.dataset.round2Save, updated_at: new Date().toISOString()};
    persist(); applyReviews(); progress(); filters();
  });
  ["round2-task", "round2-outcome", "round2-status"].forEach(id => document.getElementById(id).onchange = filters);
  document.getElementById("round2-export").onclick = () => {
    const effective = Object.fromEntries(rows.map(row => {
      const id = row.dataset.id, saved = reviews[id];
      return [id, {label: saved?.label || row.dataset.default, status: saved?.status || "accepted_initial", source: saved ? "human" : "initial_cosine"}];
    }));
    const blob = new Blob([JSON.stringify({schema_version: 1, storage_key: key, exported_at: new Date().toISOString(), reviews, effective}, null, 2)], {type: "application/json"});
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob); anchor.download = "ecot_round2_move_alignment_labels.json"; anchor.click();
    URL.revokeObjectURL(anchor.href);
  };
  document.getElementById("round2-import").onchange = event => {
    const file = event.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const payload = JSON.parse(reader.result);
      reviews = payload.reviews || payload.values?.[key] || {};
      persist(); applyReviews(); progress(); filters();
    };
    reader.readAsText(file);
  };
  applyReviews(); progress(); filters();
})();
