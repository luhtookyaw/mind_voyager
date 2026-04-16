from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIFFICULTIES = ["easy", "normal", "hard"]


def load_session_aggregates(root: Path) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for difficulty in DIFFICULTIES:
        files = sorted((root / difficulty).glob("eval_*.json"))
        rows = []
        for file_path in files:
            payload = json.loads(file_path.read_text())
            rows.append(payload["aggregate"])
        results[difficulty] = rows
    return results


def summarize(aggregates: dict[str, list[dict]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for difficulty, rows in aggregates.items():
        if not rows:
            summary[difficulty] = {
                "sessions": 0,
                "cder": {},
                "idss": {},
            }
            continue
        summary[difficulty] = {
            "sessions": len(rows),
            "cder": {
                "external": mean(r["cder"]["external"] for r in rows),
                "internal": mean(r["cder"]["internal"] for r in rows),
                "overall": mean(r["cder"]["overall"] for r in rows),
            },
            "idss": {
                "average": mean(r["idss"]["average"] for r in rows),
                "relevant_history": mean(r["idss"]["relevant_history"] for r in rows),
                "core_beliefs": mean(r["idss"]["core_beliefs"] for r in rows),
                "intermediate_beliefs": mean(r["idss"]["intermediate_beliefs"] for r in rows),
                "coping_strategies": mean(r["idss"]["coping_strategies"] for r in rows),
            },
        }
    return summary


def print_summary(summary: dict[str, dict]) -> None:
    print("Average Scores by Difficulty")
    print()
    print(f"{'Difficulty':<10} {'Sessions':>8} {'CDER-E':>10} {'CDER-I':>10} {'CDER-G':>10} {'IDSS-Avg':>10}")
    for difficulty in DIFFICULTIES:
        row = summary[difficulty]
        if row["sessions"] == 0:
            print(f"{difficulty:<10} {0:>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        print(
            f"{difficulty:<10} "
            f"{row['sessions']:>8} "
            f"{row['cder']['external'] * 100:>10.2f} "
            f"{row['cder']['internal'] * 100:>10.2f} "
            f"{row['cder']['overall'] * 100:>10.2f} "
            f"{row['idss']['average'] * 100:>10.2f}"
        )


def save_summary_json(summary: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def plot_cder(summary: dict[str, dict], output_dir: Path) -> None:
    labels = DIFFICULTIES
    ext = [summary[d]["cder"].get("external", 0.0) * 100 for d in labels]
    internal = [summary[d]["cder"].get("internal", 0.0) * 100 for d in labels]
    overall = [summary[d]["cder"].get("overall", 0.0) * 100 for d in labels]

    x = range(len(labels))
    width = 0.24

    plt.figure(figsize=(8, 5))
    plt.bar([i - width for i in x], ext, width=width, label="External")
    plt.bar(x, internal, width=width, label="Internal")
    plt.bar([i + width for i in x], overall, width=width, label="Overall")
    plt.xticks(list(x), labels)
    plt.ylabel("Score (%)")
    plt.title("CDER by Difficulty")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cder_by_difficulty.png", dpi=200)
    plt.close()


def plot_idss(summary: dict[str, dict], output_dir: Path) -> None:
    labels = DIFFICULTIES
    categories = [
        ("average", "Avg."),
        ("relevant_history", "RH"),
        ("core_beliefs", "CB"),
        ("intermediate_beliefs", "IB"),
        ("coping_strategies", "CS"),
    ]

    plt.figure(figsize=(10, 6))
    for key, label in categories:
        values = [summary[d]["idss"].get(key, 0.0) * 100 for d in labels]
        plt.plot(labels, values, marker="o", label=label)
    plt.ylabel("Score (%)")
    plt.title("IDSS by Difficulty")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "idss_by_difficulty.png", dpi=200)
    plt.close()


def plot_idss_breakdown(summary: dict[str, dict], output_dir: Path) -> None:
    labels = DIFFICULTIES
    categories = [
        ("relevant_history", "RH"),
        ("core_beliefs", "CB"),
        ("intermediate_beliefs", "IB"),
        ("coping_strategies", "CS"),
    ]
    x = range(len(labels))
    width = 0.18

    plt.figure(figsize=(10, 6))
    for idx, (key, label) in enumerate(categories):
        offset = (idx - 1.5) * width
        values = [summary[d]["idss"].get(key, 0.0) * 100 for d in labels]
        plt.bar([i + offset for i in x], values, width=width, label=label)
    plt.xticks(list(x), labels)
    plt.ylabel("Score (%)")
    plt.title("IDSS Component Breakdown by Difficulty")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "idss_breakdown_by_difficulty.png", dpi=200)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize and plot evaluation results")
    parser.add_argument(
        "--input-root",
        default="evaluations",
        help="Root directory containing easy/normal/hard evaluation files",
    )
    parser.add_argument(
        "--summary-output",
        default="evaluations/summary/averages.json",
        help="Path to save summary JSON",
    )
    parser.add_argument(
        "--plot-dir",
        default="evaluations/plots",
        help="Directory to save PNG plots",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_root = Path(args.input_root)
    summary_output = Path(args.summary_output)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    aggregates = load_session_aggregates(input_root)
    summary = summarize(aggregates)
    print_summary(summary)
    save_summary_json(summary, summary_output)
    plot_cder(summary, plot_dir)
    plot_idss(summary, plot_dir)
    plot_idss_breakdown(summary, plot_dir)

    print()
    print(f"Saved summary JSON to {summary_output}")
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
