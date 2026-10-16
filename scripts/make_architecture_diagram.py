"""Render a PNG architecture diagram for the report (Task 9).

Produces report/screenshots/architecture_diagram.png using matplotlib so it can
be embedded in a PDF/DOCX report without external tooling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "report" / "screenshots" / "architecture_diagram.png"


def box(ax, x, y, w, h, text, color):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "-|>", "color": "#555", "lw": 1.3})


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Heart Disease MLOps — Architecture", fontsize=16, pad=14)

    data_c, train_c, serve_c, deploy_c, mon_c = (
        "#DDEBF7", "#E2F0D9", "#FFF2CC", "#FCE4D6", "#EAD1DC",
    )

    # Row 1: data -> train
    box(ax, 0.3, 6.2, 2.3, 1.1, "Raw UCI data\n(processed.cleveland)", data_c)
    box(ax, 3.0, 6.2, 2.3, 1.1, "data_loader\nclean + binarize", data_c)
    box(ax, 5.7, 6.2, 2.3, 1.1, "preprocess\nimpute+scale+ohe", data_c)
    box(ax, 8.4, 6.2, 2.5, 1.1, "train\nLR+RF, CV, tuning", train_c)
    box(ax, 11.2, 6.2, 2.5, 1.1, "MLflow tracking\nparams/metrics", train_c)
    for x in [2.6, 5.3, 8.0, 10.9]:
        arrow(ax, x, 6.75, x + 0.4, 6.75)

    # model artifact
    box(ax, 8.4, 4.6, 2.5, 1.0, "model.joblib\n+ metadata", train_c)
    arrow(ax, 9.65, 6.2, 9.65, 5.6)

    # Row: CI/CD
    box(ax, 0.3, 4.6, 2.3, 1.0, "CI/CD (GH Actions)\nlint→test→train", "#D9E1F2")
    arrow(ax, 2.6, 5.1, 3.0, 5.1)
    box(ax, 3.0, 4.6, 2.3, 1.0, "26 unit tests\n+ artifacts", "#D9E1F2")

    # Serving
    box(ax, 8.4, 2.9, 2.5, 1.1, "FastAPI\n/predict /health\n/metrics", serve_c)
    arrow(ax, 9.65, 4.6, 9.65, 4.0)
    box(ax, 11.2, 2.9, 2.5, 1.1, "Docker image\nnon-root+health", serve_c)
    arrow(ax, 10.9, 3.45, 11.2, 3.45)

    # Deployment
    box(ax, 8.4, 1.1, 2.5, 1.1, "K8s Deployment\n2 replicas+probes", deploy_c)
    arrow(ax, 12.45, 2.9, 12.45, 1.9)
    arrow(ax, 12.45, 1.9, 10.9, 1.65)
    box(ax, 11.2, 1.1, 2.5, 1.1, "LoadBalancer\n/ Ingress", deploy_c)
    arrow(ax, 9.65, 2.9, 9.65, 2.2)
    box(ax, 5.7, 1.1, 2.3, 1.1, "Client / curl", deploy_c)
    arrow(ax, 11.2, 1.65, 8.0, 1.65)

    # Monitoring
    box(ax, 0.3, 2.9, 2.3, 1.1, "Prometheus\nscrape /metrics", mon_c)
    arrow(ax, 8.4, 3.45, 2.6, 3.45)
    box(ax, 0.3, 1.1, 2.3, 1.1, "Grafana\ndashboard", mon_c)
    arrow(ax, 1.45, 2.9, 1.45, 2.2)

    # legend
    handles = [
        mpatches.Patch(color=data_c, label="Data / Features"),
        mpatches.Patch(color=train_c, label="Training / Tracking"),
        mpatches.Patch(color="#D9E1F2", label="CI/CD"),
        mpatches.Patch(color=serve_c, label="Serving"),
        mpatches.Patch(color=deploy_c, label="Deployment"),
        mpatches.Patch(color=mon_c, label="Monitoring"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=6,
              bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture diagram -> {OUT}")


if __name__ == "__main__":
    main()
