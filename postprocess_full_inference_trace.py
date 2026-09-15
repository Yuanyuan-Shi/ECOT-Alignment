#!/usr/bin/env python3
"""Correlate CUDA matmul kernels with CPU tensor shapes and model-context intervals."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.trace, "rt") as handle:
        payload = json.load(handle)
    events = payload["traceEvents"]
    contexts = []
    cpu_by_external = {}
    for event in events:
        cat = event.get("cat")
        name = event.get("name", "")
        if cat == "user_annotation" and name.startswith("MODEL_CONTEXT::"):
            contexts.append((event["ts"], event["ts"] + event.get("dur", 0), name.split("::", 1)[1]))
        elif cat == "cpu_op" and "External id" in event.get("args", {}):
            cpu_by_external[event["args"]["External id"]] = event
    contexts.sort()
    starts = [row[0] for row in contexts]

    def context_at(timestamp):
        index = bisect.bisect_right(starts, timestamp) - 1
        while index >= 0:
            start, end, label = contexts[index]
            if end >= timestamp:
                return label
            # Context intervals are local and nested; stop after a bounded backwards search.
            if index < bisect.bisect_right(starts, timestamp) - 64:
                break
            index -= 1
        return None

    tokens = ("gemm", "gemv", "matmul", "cutlass", "mma")
    rows = []
    kernel_counts = Counter()
    context_counts = Counter()
    for event in events:
        if event.get("cat") != "kernel" or not any(token in event.get("name", "").lower() for token in tokens):
            continue
        external_id = event.get("args", {}).get("External id")
        cpu = cpu_by_external.get(external_id, {})
        cpu_args = cpu.get("args", {})
        context = context_at(cpu.get("ts", event["ts"]))
        row = {
            "kernel": event["name"],
            "duration_us": event.get("dur"),
            "correlation": event.get("args", {}).get("correlation"),
            "external_id": external_id,
            "cpu_operator": cpu.get("name"),
            "input_dims": cpu_args.get("Input Dims"),
            "input_types": cpu_args.get("Input type"),
            "input_strides": cpu_args.get("Input Strides"),
            "model_context": context,
        }
        rows.append(row)
        kernel_counts[row["kernel"]] += 1
        context_counts[context or "unclassified"] += 1
    result = {
        "source_trace": args.trace.name,
        "context_intervals": len(contexts),
        "correlated_cuda_matmul_events": len(rows),
        "classified_events": sum(value for key, value in context_counts.items() if key != "unclassified"),
        "unclassified_events": context_counts["unclassified"],
        "kernel_counts": [{"kernel": key, "count": value} for key, value in kernel_counts.most_common()],
        "model_context_counts": [{"model_context": key, "count": value} for key, value in context_counts.most_common()],
        "events": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "source_trace", "context_intervals", "correlated_cuda_matmul_events", "classified_events", "unclassified_events"
    )}, indent=2))


if __name__ == "__main__":
    main()
