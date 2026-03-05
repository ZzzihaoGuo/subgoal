#!/usr/bin/env python3
"""Sweep ATACOM manifold parameters and save results to CSV + markdown."""

import subprocess
import csv
import re
import itertools
from datetime import datetime

# ===== Sweep 参数配置 =====
# 每个参数: (flag, values_list)
# 固定参数用单个值, 要 sweep 的参数给多个值
SWEEP_PARAMS = {
    "topk":         [3, 5, 8],
    "viab_gain":    [0.5, 1.0],
    "err_gain":     [5.0, 10.0],
    "alpha_max":    [1.0],           # 固定
    "g_act_thresh": [0.01],          # 固定
    "safety_margin":[0.02],          # 固定
    "n_lookahead":  [0, 2],
    "w_slack":      [1.0, 5.0, 10.0],
}

# 固定的其他参数
FIXED_ARGS = [
    "--no-video",
    "--epi", "300",       # 每组 300 episodes, 平衡速度和统计量
]

# 从 test_manifold.py 的输出中提取结果的正则
RESULT_PATTERN = re.compile(
    r"last_reward:\s*([-\d.]+),\s*last_dist:\s*([-\d.]+),\s*"
    r"safe_rate:\s*([-\d.]+)%,\s*"
    r"success_agent:\s*([-\d.]+)%,\s*success_epi:\s*([-\d.]+)%"
)

def run_single(params: dict) -> dict:
    """Run test_manifold.py with given params and parse results."""
    cmd = ["python", "test_manifold.py"] + FIXED_ARGS
    for key, val in params.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(val)])

    param_str = " ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{'='*60}")
    print(f"Running: {param_str}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd="/home/a5l/zihao1996.a5l/project/subgoal"
        )
        output = result.stdout + result.stderr

        # Find the last summary line
        match = None
        for line in output.split("\n"):
            m = RESULT_PATTERN.search(line)
            if m:
                match = m

        if match:
            return {
                "last_reward": float(match.group(1)),
                "last_dist": float(match.group(2)),
                "safe_rate": float(match.group(3)),
                "success_agent": float(match.group(4)),
                "success_epi": float(match.group(5)),
                "status": "ok",
            }
        else:
            # Print last 20 lines for debugging
            lines = output.strip().split("\n")
            print("  [WARN] Could not parse results. Last lines:")
            for l in lines[-20:]:
                print(f"    {l}")
            return {"status": "parse_error"}

    except subprocess.TimeoutExpired:
        print("  [WARN] Timeout!")
        return {"status": "timeout"}
    except Exception as e:
        print(f"  [WARN] Error: {e}")
        return {"status": f"error: {e}"}


def main():
    # Generate all combinations
    keys = list(SWEEP_PARAMS.keys())
    values = list(SWEEP_PARAMS.values())
    combos = list(itertools.product(*values))

    print(f"Total combinations: {len(combos)}")
    print(f"Estimated time: ~{len(combos) * 2} minutes (depends on --epi)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"sweep_results_{timestamp}.csv"
    md_path = f"sweep_results_{timestamp}.md"

    # CSV header
    fieldnames = keys + ["safe_rate", "last_dist", "last_reward", "success_agent", "success_epi", "status"]

    results = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            print(f"\n[{i+1}/{len(combos)}]")

            res = run_single(params)

            row = {**params, **res}
            writer.writerow(row)
            f.flush()  # 实时写入, 中断也不丢数据
            results.append(row)

            if res["status"] == "ok":
                print(f"  => safe_rate={res['safe_rate']:.1f}%, last_dist={res['last_dist']:.4f}, "
                      f"success_epi={res['success_epi']:.1f}%")

    # Generate markdown summary
    # Sort by safe_rate desc, then last_dist asc
    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results.sort(key=lambda r: (-r["safe_rate"], r["last_dist"]))

    with open(md_path, "w") as f:
        f.write(f"# ATACOM Manifold Parameter Sweep Results\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**Episodes per config**: {FIXED_ARGS[FIXED_ARGS.index('--epi') + 1]}\n\n")

        # Best results
        f.write("## Top 10 (by safe_rate, then last_dist)\n\n")
        f.write("| # | topk | K | Kc | lookahead | w_slack | safe_rate | last_dist | success_epi |\n")
        f.write("|---|------|---|----|-----------|---------|-----------|-----------|-------------|\n")
        for i, r in enumerate(ok_results[:10]):
            f.write(f"| {i+1} | {r['topk']} | {r['viab_gain']} | {r['err_gain']} | "
                    f"{r['n_lookahead']} | {r['w_slack']} | "
                    f"**{r['safe_rate']:.1f}%** | {r['last_dist']:.4f} | {r['success_epi']:.1f}% |\n")

        # Results meeting target
        f.write("\n## Configs meeting targets (safe_rate >= 99.5%, last_dist <= 0.03)\n\n")
        target = [r for r in ok_results if r["safe_rate"] >= 99.5 and r["last_dist"] <= 0.03]
        if target:
            f.write("| topk | K | Kc | lookahead | w_slack | safe_rate | last_dist | success_epi |\n")
            f.write("|------|---|----|-----------|---------|-----------|-----------|-------------|\n")
            for r in target:
                f.write(f"| {r['topk']} | {r['viab_gain']} | {r['err_gain']} | "
                        f"{r['n_lookahead']} | {r['w_slack']} | "
                        f"**{r['safe_rate']:.1f}%** | {r['last_dist']:.4f} | {r['success_epi']:.1f}% |\n")
        else:
            f.write("*No configs met both targets.*\n")

        # Full results table
        f.write("\n## All Results\n\n")
        f.write("| topk | K | Kc | lookahead | w_slack | safe_rate | last_dist | success_epi | status |\n")
        f.write("|------|---|----|-----------|---------|-----------|-----------|-------------|--------|\n")
        for r in results:
            if r.get("status") == "ok":
                f.write(f"| {r['topk']} | {r['viab_gain']} | {r['err_gain']} | "
                        f"{r['n_lookahead']} | {r['w_slack']} | "
                        f"{r['safe_rate']:.1f}% | {r['last_dist']:.4f} | {r['success_epi']:.1f}% | ok |\n")
            else:
                f.write(f"| {r['topk']} | {r.get('viab_gain','')} | {r.get('err_gain','')} | "
                        f"{r.get('n_lookahead','')} | {r.get('w_slack','')} | "
                        f"- | - | - | {r.get('status','')} |\n")

    print(f"\n{'='*60}")
    print(f"Sweep complete! {len(ok_results)}/{len(combos)} succeeded.")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")

    if ok_results:
        best = ok_results[0]
        print(f"\nBest: topk={best['topk']}, K={best['viab_gain']}, Kc={best['err_gain']}, "
              f"lookahead={best['n_lookahead']}, w_slack={best['w_slack']}")
        print(f"  safe_rate={best['safe_rate']:.1f}%, last_dist={best['last_dist']:.4f}")


if __name__ == "__main__":
    main()
