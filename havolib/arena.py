"""
Benchmark Arena — public leaderboard for regime-shift detection methods.

Auto-runs HAVOK vs all baselines on 20+ public datasets, generates
a JSON leaderboard, and prints rankings.

This is the Phase 2 feature — become THE standard for comparison.

Usage:
    python -m havolib.arena --datasets all --output leaderboard.json
"""

from __future__ import annotations
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger("havok.arena")


@dataclass
class ArenaEntry:
    """One entry in the leaderboard."""
    rank: int
    method: str
    dataset: str
    detection_delay: int
    separation_score: float
    false_positives: int
    compute_time_ms: float
    timestamp: str = ""


class BenchmarkArena:
    """Public benchmark arena — run all methods on all datasets."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.entries: List[ArenaEntry] = []

    def run(self, methods: Optional[List[str]] = None, verbose: bool = True) -> List[ArenaEntry]:
        """Run full benchmark arena.

        Args:
            methods: list of method names (None = all)
            verbose: print progress

        Returns:
            list of ArenaEntry sorted by rank
        """
        from havolib.benchmark.runner import run_benchmark, print_summary

        if methods is None:
            methods = ["havok_pro", "havok_basic", "rolling_std", "cusum", "arima_residual"]

        results = run_benchmark(methods=methods, verbose=verbose)
        print_summary(results)

        # Convert to arena entries
        rank = 1
        for ds_name, ds_result in results.items():
            for m_name, m_result in ds_result.methods.items():
                entry = ArenaEntry(
                    rank=rank,
                    method=m_name,
                    dataset=ds_name,
                    detection_delay=m_result.detection_delay,
                    separation_score=m_result.separation_score,
                    false_positives=m_result.false_positives,
                    compute_time_ms=m_result.compute_time_ms,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                self.entries.append(entry)
                rank += 1

        # Sort by separation score
        self.entries.sort(key=lambda e: e.separation_score, reverse=True)
        for i, e in enumerate(self.entries):
            e.rank = i + 1

        return self.entries

    def save_leaderboard(self, filename: str = "leaderboard.json") -> str:
        """Save leaderboard as JSON."""
        path = self.output_dir / filename
        data = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_entries": len(self.entries),
            "entries": [
                {
                    "rank": e.rank,
                    "method": e.method,
                    "dataset": e.dataset,
                    "detection_delay": e.detection_delay,
                    "separation_score": round(e.separation_score, 4),
                    "false_positives": e.false_positives,
                    "compute_time_ms": round(e.compute_time_ms, 1),
                }
                for e in self.entries
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Leaderboard saved to {path}")
        return str(path)

    def print_leaderboard(self) -> str:
        """Print formatted leaderboard."""
        lines = [
            "=" * 70,
            "🏆 HAVOK BENCHMARK ARENA — PUBLIC LEADERBOARD",
            "=" * 70,
            f"{'Rank':<5} {'Method':<22} {'Dataset':<18} {'Separation':>10} {'Delay':>7} {'FP':>5} {'Time':>8}",
            "-" * 70,
        ]
        for e in self.entries[:20]:
            delay = str(e.detection_delay) if e.detection_delay >= 0 else "N/D"
            lines.append(
                f"{e.rank:<5} {e.method:<22} {e.dataset:<18} "
                f"{e.separation_score:>10.4f} {delay:>7} {e.false_positives:>5} "
                f"{e.compute_time_ms:>7.0f}ms"
            )
        lines.append("=" * 70)

        # Top methods summary
        from collections import defaultdict
        method_avgs = defaultdict(list)
        for e in self.entries:
            method_avgs[e.method].append(e.separation_score)

        lines.append("\n📊 METHOD RANKING (avg separation):")
        for name, scores in sorted(method_avgs.items(), key=lambda x: np.mean(x[1]), reverse=True):
            lines.append(f"  {name:<22} {np.mean(scores):.4f} ({len(scores)} datasets)")

        return "\n".join(lines)


def run_arena(output: str = "arena_results/leaderboard.json", quiet: bool = False):
    """Convenience function: run arena and save leaderboard."""
    arena = BenchmarkArena()
    arena.run(verbose=not quiet)
    arena.save_leaderboard(output)
    if not quiet:
        print(arena.print_leaderboard())
    return arena
