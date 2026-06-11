#!/usr/bin/env python3
"""
GNN 性能对比脚本：GraphSAGE vs GAT

改进点：
  - 异构图只建一次（shared/），两个模型共用，确保公平比较
  - 修正项目路径：Path(__file__).resolve().parent.parent.parent
  - 7 张学术黑白图（柱状图、损失曲线、AUC/F1/AP 曲线、雷达图、PR 散点+iso-F1）
  - JSON 报告 + 控制台对比表（含 Delta 行）

用法：
  python SAGE_GAT.py [--epochs 50] [--device auto] [--processed_dir <path>]
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无头环境
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

# ── 项目根目录（performance_vs -> src -> fraud-detection-gnn）────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import run_graph_building, run_model_training, setup_logging, load_config
from loguru import logger

# ── 输出根目录 ────────────────────────────────────────────────────────────────
OUTPUT_BASE = str(PROJECT_ROOT / "outputs" / "comparison")

EXPERIMENTS = [
    {"name": "GraphSAGE", "type": "graphsage"},
    {"name": "GAT",       "type": "gat"},
]

# ── Matplotlib 黑白学术风格 ───────────────────────────────────────────────────
BW_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.2,
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.6,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "lines.linewidth":   1.8,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "black",
}

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS     = ["o", "s", "^", "D"]
HATCHES     = ["/", "\\"]
COLORS      = ["white", "#888888"]    # 白色 / 灰色，黑色边框


# ── Mock Args（模拟 argparse.Namespace） ──────────────────────────────────────
class _Args:
    def __init__(self, model_type: str, output_dir: str,
                 epochs: int, device: str, processed_dir: str):
        self.config        = str(PROJECT_ROOT / "config" / "config.yaml")
        self.data_dir      = processed_dir
        self.output_dir    = output_dir
        self.model         = model_type
        self.epochs        = epochs
        self.device        = device
        self.skip_spark    = True
        self.skip_training = False
        self.use_hybrid    = False


# ── Step 1：建图（只跑一次，两个模型共用） ────────────────────────────────────
def build_shared_graph(processed_dir: str, epochs: int, device: str):
    """建图并返回 (data, edges_df, config)。
    优先复用 outputs/graph/fraud_graph.pt（已存在则跳过重建）。"""
    # 直接用项目 outputs/ 目录，让 run_graph_building 自动找到已有的 graph
    shared_dir = str(PROJECT_ROOT / "outputs")
    os.makedirs(shared_dir, exist_ok=True)
    setup_logging(os.path.join(OUTPUT_BASE, "logs"))

    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    args   = _Args("graphsage", shared_dir, epochs, device, processed_dir)

    print(f"\n{'='*60}")
    print("  Step 1/3  建立共享异构图（card-transact-merchant）")
    print(f"  数据目录: {processed_dir}")
    print(f"{'='*60}")

    data, edges_df = run_graph_building(processed_dir, args)
    return data, edges_df, config


# ── Step 2：训练单个模型 ──────────────────────────────────────────────────────
def run_experiment(exp: dict, data, config: dict,
                   epochs: int, device: str) -> dict:
    """训练一个 GNN 模型，读取结果 JSON，返回结果 dict"""
    model_name = exp["name"]
    model_type = exp["type"]
    out_dir    = os.path.join(OUTPUT_BASE, model_type)
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(os.path.join(out_dir, "logs"))

    # processed_dir 对训练阶段已不需要，传空字符串即可
    args = _Args(model_type, out_dir, epochs, device, "")

    print(f"\n{'='*60}")
    print(f"  训练模型: {model_name}  ({epochs} epochs)")
    print(f"  输出目录: {out_dir}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        run_model_training(data, args, config)
        elapsed = time.time() - t0

        results_json = os.path.join(
            out_dir, "models",
            f"{model_type}_fraud_detector_results.json"
        )
        with open(results_json, "r", encoding="utf-8") as f:
            res = json.load(f)

        history = res.get("train", {}).get("history", {})

        print(f"  [OK]  AUC={res['test']['auc']:.4f}  "
              f"AP={res['test']['ap']:.4f}  "
              f"F1={res['test']['f1']:.4f}  "
              f"耗时={elapsed:.0f}s")

        return {
            "name":         model_name,
            "type":         model_type,
            "status":       "success",
            "elapsed":      elapsed,
            "test":         res["test"],
            "best_val_auc": res["train"].get("best_val_auc", 0.0),
            "history":      history,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {model_name}: {exc}")
        return {
            "name":    model_name,
            "type":    model_type,
            "status":  "failed",
            "error":   str(exc),
            "elapsed": elapsed,
        }


# ── 控制台对比表 ──────────────────────────────────────────────────────────────
def print_comparison(results: list):
    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        print("\n所有实验均失败，无对比数据。")
        return

    HDR = (f"{'Model':<14}{'AUC':>8}{'AP':>8}{'F1':>8}"
           f"{'Prec':>8}{'Recall':>8}{'Time(s)':>9}")
    SEP = "-" * len(HDR)

    print(f"\n{'模型对比结果':^{len(HDR)}}")
    print(SEP)
    print(HDR)
    print(SEP)

    for r in ok:
        t = r["test"]
        print(f"{r['name']:<14}"
              f"{t['auc']:>8.4f}"
              f"{t['ap']:>8.4f}"
              f"{t['f1']:>8.4f}"
              f"{t['precision']:>8.4f}"
              f"{t['recall']:>8.4f}"
              f"{r['elapsed']:>9.1f}")

    if len(ok) == 2:
        d  = {k: ok[0]["test"][k] - ok[1]["test"][k]
              for k in ("auc", "ap", "f1", "precision", "recall")}
        dt = ok[0]["elapsed"] - ok[1]["elapsed"]
        print(SEP)
        print(f"{'Delta (0-1)':<14}"
              f"{d['auc']:>+8.4f}"
              f"{d['ap']:>+8.4f}"
              f"{d['f1']:>+8.4f}"
              f"{d['precision']:>+8.4f}"
              f"{d['recall']:>+8.4f}"
              f"{dt:>+9.1f}")

    print(SEP)
    best = max(ok, key=lambda r: r["test"]["auc"])
    print(f"\n  最佳模型（AUC）: {best['name']}  "
          f"AUC={best['test']['auc']:.4f}")


# ── 7 张学术黑白图 ────────────────────────────────────────────────────────────
def plot_all(results: list):
    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        return

    plt.rcParams.update(BW_STYLE)
    plot_dir = os.path.join(OUTPUT_BASE, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── 图1: 综合指标柱状图 ───────────────────────────────────────────────────
    metric_keys   = ["auc", "ap", "f1", "precision", "recall"]
    metric_labels = ["AUC", "AP", "F1", "Precision", "Recall"]
    x = np.arange(len(metric_keys))
    w = 0.35
    n = len(ok)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(ok):
        vals   = [r["test"][m] for m in metric_keys]
        offset = (i - (n - 1) / 2) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=r["name"],
                        color=COLORS[i % 2],
                        edgecolor="black",
                        hatch=HATCHES[i % 2],
                        linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("GraphSAGE vs GAT — Test Metrics Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "01_metrics_bar.png"), dpi=150)
    plt.close(fig)

    # ── 图2: 训练损失曲线 ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(ok):
        loss = r.get("history", {}).get("train_loss", [])
        if loss:
            n_pts = len(loss)
            ax.plot(loss,
                    label=r["name"],
                    linestyle=LINE_STYLES[i % 4],
                    marker=MARKERS[i % 4],
                    markevery=max(1, n_pts // 10),
                    markersize=5,
                    color="black" if i == 0 else "#555555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "02_train_loss.png"), dpi=150)
    plt.close(fig)

    # ── 图3: 验证 AUC 曲线 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(ok):
        curve = r.get("history", {}).get("val_auc", [])
        if curve:
            ax.plot(curve,
                    label=r["name"],
                    linestyle=LINE_STYLES[i % 4],
                    marker=MARKERS[i % 4],
                    markevery=max(1, len(curve) // 10),
                    markersize=5,
                    color="black" if i == 0 else "#555555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "03_val_auc.png"), dpi=150)
    plt.close(fig)

    # ── 图4: 验证 F1 曲线 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(ok):
        curve = r.get("history", {}).get("val_f1", [])
        if curve:
            ax.plot(curve,
                    label=r["name"],
                    linestyle=LINE_STYLES[i % 4],
                    marker=MARKERS[i % 4],
                    markevery=max(1, len(curve) // 10),
                    markersize=5,
                    color="black" if i == 0 else "#555555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Validation F1 Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "04_val_f1.png"), dpi=150)
    plt.close(fig)

    # ── 图5: 验证 AP 曲线 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(ok):
        curve = r.get("history", {}).get("val_ap", [])
        if curve:
            ax.plot(curve,
                    label=r["name"],
                    linestyle=LINE_STYLES[i % 4],
                    marker=MARKERS[i % 4],
                    markevery=max(1, len(curve) // 10),
                    markersize=5,
                    color="black" if i == 0 else "#555555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AP")
    ax.set_title("Validation Average Precision Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "05_val_ap.png"), dpi=150)
    plt.close(fig)

    # ── 图6: 雷达图 ──────────────────────────────────────────────────────────
    cats   = ["AUC", "AP", "F1", "Precision", "Recall"]
    mkeys  = ["auc", "ap", "f1", "precision", "recall"]
    N      = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # 闭合

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, r in enumerate(ok):
        vals = [r["test"][k] for k in mkeys] + [r["test"][mkeys[0]]]
        ax.plot(angles, vals,
                label=r["name"],
                linewidth=2.0 if i == 0 else 1.5,
                linestyle=LINE_STYLES[i % 4],
                color="black" if i == 0 else "#555555")
        ax.fill(angles, vals,
                alpha=0.08 if i == 0 else 0.12,
                color="black" if i == 0 else "#888888")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Radar Chart", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "06_radar.png"), dpi=150)
    plt.close(fig)

    # ── 图7: Precision-Recall 操作点 + iso-F1 等值线 ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # iso-F1 等值线
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        p_range = np.linspace(f1_val / 2 + 1e-6, 1.0, 300)
        r_range = f1_val * p_range / (2 * p_range - f1_val + 1e-9)
        valid   = (r_range >= 0) & (r_range <= 1)
        ax.plot(r_range[valid], p_range[valid],
                color="#bbbbbb", linewidth=0.8, linestyle="--")
        idx = len(p_range[valid]) // 3
        if idx < len(p_range[valid]):
            ax.text(r_range[valid][idx], p_range[valid][idx],
                    f"F1={f1_val}", fontsize=7, color="#888888")

    mks = ["o", "s"]
    for i, r in enumerate(ok):
        ax.scatter(r["test"]["recall"], r["test"]["precision"],
                   marker=mks[i % 2], s=120,
                   color="black" if i == 0 else "white",
                   edgecolors="black", linewidths=1.5,
                   zorder=5, label=r["name"])
        ax.annotate(
            f"  {r['name']}\n  AUC={r['test']['auc']:.4f}",
            (r["test"]["recall"], r["test"]["precision"]),
            fontsize=9
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision–Recall Operating Point\n(with iso-F1 contours)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "07_pr_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\n  7 张图已保存至: {plot_dir}")


# ── 保存 JSON 报告 ────────────────────────────────────────────────────────────
def save_report(results: list) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_BASE, f"comparison_{ts}.json")
    ok   = [r for r in results if r["status"] == "success"]

    report = {
        "timestamp": datetime.now().isoformat(),
        "results":   results,
        "summary": {
            "total":   len(results),
            "success": len(ok),
            "failed":  len(results) - len(ok),
        },
    }

    if ok:
        report["comparison_table"] = [
            {
                "model":     r["name"],
                "test_auc":  r["test"]["auc"],
                "test_ap":   r["test"]["ap"],
                "test_f1":   r["test"]["f1"],
                "precision": r["test"]["precision"],
                "recall":    r["test"]["recall"],
                "time_s":    r["elapsed"],
            }
            for r in ok
        ]
        best = max(ok, key=lambda r: r["test"]["auc"])
        report["best_model"] = best["name"]

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  JSON 报告已保存: {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="GraphSAGE vs GAT 性能对比（共享图，公平对比）"
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="训练轮数（默认 50）"
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="cpu / cuda / auto"
    )
    p.add_argument(
        "--processed_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="预处理数据目录（含 edges.parquet / card_features.parquet 等）"
    )
    return p.parse_args()


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  GNN 性能对比实验: GraphSAGE  vs  GAT")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Device  : {args.device}")
    print(f"  数据目录: {args.processed_dir}")
    print(f"  输出根目录: {OUTPUT_BASE}")
    print(f"{'='*60}")

    # Step 1: 共享图（建一次）
    data, _, config = build_shared_graph(
        args.processed_dir, args.epochs, args.device
    )

    # Step 2: 逐个训练
    results = []
    for exp in EXPERIMENTS:
        r = run_experiment(exp, data, config, args.epochs, args.device)
        results.append(r)

    # Step 3: 汇总输出
    print_comparison(results)
    plot_all(results)
    save_report(results)

    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"  输出目录: {OUTPUT_BASE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
