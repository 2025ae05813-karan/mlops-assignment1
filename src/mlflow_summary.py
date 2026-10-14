"""Export a human-readable summary of all MLflow runs (Assignment Task 3).

Produces report/experiment_tracking_summary.md so the experiment history is
captured in the written report even without a live UI screenshot.

Usage:
    python -m src.mlflow_summary
"""
from __future__ import annotations

import logging

import mlflow

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = config.REPORT_DIR / "experiment_tracking_summary.md"


def export_summary() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_DIR.as_uri())
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        raise RuntimeError(
            "No MLflow experiment found. Run `python -m src.train --mlflow` first."
        )

    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time ASC"])

    lines: list[str] = []
    lines.append("# Experiment Tracking Summary (MLflow)\n")
    lines.append(f"**Experiment:** `{exp.name}`  ")
    lines.append(f"**Tracking URI:** `{config.MLFLOW_TRACKING_DIR.as_uri()}`  ")
    lines.append(f"**Total runs:** {len(runs)}\n")
    lines.append("Launch the UI with:\n")
    lines.append("```bash\nmlflow ui --backend-store-uri ./mlruns\n```\n")

    for r in runs:
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        lines.append(f"## Run: `{name}`\n")
        lines.append(f"- **Run ID:** `{r.info.run_id}`")
        lines.append(f"- **Status:** {r.info.status}")

        lines.append("\n**Parameters**\n")
        lines.append("| param | value |")
        lines.append("|-------|-------|")
        for k, v in sorted(r.data.params.items()):
            lines.append(f"| {k} | {v} |")

        lines.append("\n**Metrics**\n")
        lines.append("| metric | value |")
        lines.append("|--------|-------|")
        for k, v in sorted(r.data.metrics.items()):
            lines.append(f"| {k} | {v:.4f} |")

        arts = [a.path for a in client.list_artifacts(r.info.run_id)]
        lines.append("\n**Logged artifacts:** " + ", ".join(f"`{a}`" for a in arts) + "\n")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines))
    logger.info("Wrote experiment tracking summary -> %s", OUTPUT)


if __name__ == "__main__":
    export_summary()
