"""Deduplicate overlapping ranked candidates and keep the best recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from windows import format_timestamp


def overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    shorter = min(left["end"] - left["start"], right["end"] - right["start"])
    return overlap / shorter if shorter > 0 else 0.0


def select_top(raw_scores: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
    candidates = [score for score in raw_scores["scores"] if score["candidate"] and score["overall_score"] >= 50 and score["end"] > score["start"]]
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda value: value["overall_score"], reverse=True):
        if all(overlap_ratio(candidate, chosen) < 0.6 for chosen in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def write_selections(raw_scores_path: Path, output_path: Path, top_k: int) -> dict[str, Any]:
    raw_scores = json.loads(raw_scores_path.read_text(encoding="utf-8"))
    recommendations = []
    for score in select_top(raw_scores, top_k):
        recommendations.append({
            **score,
            "start_timestamp": format_timestamp(score["start"]),
            "end_timestamp": format_timestamp(score["end"]),
        })
    data = {"raw_scores": str(raw_scores_path), "top_k": top_k, "recommendations": recommendations}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
