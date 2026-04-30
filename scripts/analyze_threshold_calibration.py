from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DIFFICULTIES = ("easy", "normal", "hard")
FIELDS = (
    "situation",
    "automatic_thought",
    "emotion",
    "behavior",
    "relevant_history",
    "core_beliefs",
    "intermediate_beliefs",
    "coping_strategies",
)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def load_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        difficulty_dir = root / difficulty
        if not difficulty_dir.exists():
            continue
        for session_path in sorted(difficulty_dir.glob("session_*.json")):
            payload = json.loads(session_path.read_text())
            case_id = payload["case_id"]
            for event in payload.get("turn_similarity_events", []):
                row = dict(event)
                row["difficulty"] = difficulty
                row["case_id"] = case_id
                row["session_path"] = str(session_path)
                events.append(row)
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        difficulty: {field: [] for field in FIELDS} for difficulty in DIFFICULTIES
    }
    for event in events:
        grouped[event["difficulty"]][event["field"]].append(event)

    summary: dict[str, dict[str, dict[str, Any]]] = {
        difficulty: {} for difficulty in DIFFICULTIES
    }
    for difficulty in DIFFICULTIES:
        for field in FIELDS:
            rows = grouped[difficulty][field]
            scored = [row for row in rows if row["score"] is not None]
            scores = [float(row["score"]) for row in scored]
            revealed_rows = [row for row in rows if row["revealed"]]
            threshold_hits = [
                row for row in scored if float(row["score"]) >= float(row["threshold"])
            ]
            blocked = [row for row in rows if not row["prerequisites_met"]]

            summary[difficulty][field] = {
                "events": len(rows),
                "scored_events": len(scored),
                "reveals": len(revealed_rows),
                "reveal_rate": (len(revealed_rows) / len(rows)) if rows else None,
                "threshold_hits": len(threshold_hits),
                "blocked_by_prereq": len(blocked),
                "mean_score": mean(scores) if scores else None,
                "median_score": median(scores) if scores else None,
                "p75_score": percentile(scores, 0.75),
                "p90_score": percentile(scores, 0.90),
                "max_score": max(scores) if scores else None,
            }
    return summary


def format_summary(summary: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for difficulty in DIFFICULTIES:
        lines.append(f"\n=== {difficulty.upper()} ===")
        lines.append(
            f"{'Field':<24} {'N':>5} {'Scored':>7} {'Reveals':>8} "
            f"{'Rate':>7} {'Hits':>6} {'Blocked':>8} "
            f"{'Mean':>7} {'P75':>7} {'P90':>7} {'Max':>7}"
        )
        for field in FIELDS:
            stats = summary[difficulty][field]
            rate = "-" if stats["reveal_rate"] is None else f"{stats['reveal_rate']:.2f}"
            mean_score = "-" if stats["mean_score"] is None else f"{stats['mean_score']:.3f}"
            p75_score = "-" if stats["p75_score"] is None else f"{stats['p75_score']:.3f}"
            p90_score = "-" if stats["p90_score"] is None else f"{stats['p90_score']:.3f}"
            max_score = "-" if stats["max_score"] is None else f"{stats['max_score']:.3f}"
            lines.append(
                f"{field:<24} {stats['events']:>5} {stats['scored_events']:>7} "
                f"{stats['reveals']:>8} {rate:>7} {stats['threshold_hits']:>6} "
                f"{stats['blocked_by_prereq']:>8} {mean_score:>7} "
                f"{p75_score:>7} {p90_score:>7} {max_score:>7}"
            )
    return "\n".join(lines)


def format_examples(
    events: list[dict[str, Any]],
    limit: int,
    near_margin: float,
) -> str:
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_field[event["field"]].append(event)

    lines: list[str] = []
    for field in FIELDS:
        scored = [row for row in by_field[field] if row["score"] is not None]
        highest = sorted(scored, key=lambda row: float(row["score"]), reverse=True)[:limit]
        near = [
            row
            for row in scored
            if not row["revealed"]
            and float(row["threshold"]) - near_margin <= float(row["score"]) < float(row["threshold"])
        ]
        near = sorted(near, key=lambda row: float(row["score"]), reverse=True)[:limit]

        lines.append(f"\n=== FIELD EXAMPLES: {field} ===")
        lines.append("Top scoring questions:")
        if not highest:
            lines.append("- none")
        for row in highest:
            lines.append(
                f"- [{row['difficulty']}] case={row['case_id']} turn={row['turn']} "
                f"score={float(row['score']):.3f} threshold={float(row['threshold']):.3f} "
                f"revealed={row['revealed']}"
            )
            lines.append(f"  Q: {row['therapist_text']}")

        lines.append("Near-threshold non-reveals:")
        if not near:
            lines.append("- none")
        for row in near:
            lines.append(
                f"- [{row['difficulty']}] case={row['case_id']} turn={row['turn']} "
                f"score={float(row['score']):.3f} threshold={float(row['threshold']):.3f}"
            )
            lines.append(f"  Q: {row['therapist_text']}")
    return "\n".join(lines)


def build_json_report(
    events: list[dict[str, Any]],
    summary: dict[str, dict[str, dict[str, Any]]],
    limit: int,
    near_margin: float,
) -> dict[str, Any]:
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_field[event["field"]].append(event)

    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for field in FIELDS:
        scored = [row for row in by_field[field] if row["score"] is not None]
        highest = sorted(scored, key=lambda row: float(row["score"]), reverse=True)[:limit]
        near = [
            row
            for row in scored
            if not row["revealed"]
            and float(row["threshold"]) - near_margin <= float(row["score"]) < float(row["threshold"])
        ]
        near = sorted(near, key=lambda row: float(row["score"]), reverse=True)[:limit]
        examples[field] = {
            "top_scoring": highest,
            "near_threshold_non_reveals": near,
        }

    return {
        "summary": summary,
        "examples": examples,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze threshold-calibration session logs and summarize per-field score distributions."
    )
    parser.add_argument(
        "--input-root",
        default="outputs/threshold_calibration",
        help="Root directory containing easy/normal/hard calibration session JSON files.",
    )
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=5,
        help="Number of top and near-threshold example questions to print per field.",
    )
    parser.add_argument(
        "--near-margin",
        type=float,
        default=0.05,
        help="Score margin below threshold to treat as a near-threshold non-reveal.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the text report.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save the structured JSON report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.input_root)
    events = load_events(root)
    if not events:
        raise SystemExit(f"No turn_similarity_events found under {root}")

    summary = summarize(events)
    text_report = "\n".join(
        [
            format_summary(summary),
            format_examples(events, limit=args.examples_limit, near_margin=args.near_margin),
        ]
    )
    print(text_report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text_report)

    if args.json_output:
        json_output_path = Path(args.json_output)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_report = build_json_report(
            events,
            summary,
            limit=args.examples_limit,
            near_margin=args.near_margin,
        )
        json_output_path.write_text(json.dumps(json_report, indent=2))


if __name__ == "__main__":
    main()
