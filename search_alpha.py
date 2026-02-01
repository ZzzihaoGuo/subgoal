"""Grid search for optimal CBF alpha1, alpha2 values."""
import subprocess
import re
import os
import itertools
import argparse
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="GPU device id to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# === Configuration ===
PATH = "logs/LidarSpread/informarl_subgoal/seed0_129154248_ZFZO"
EPI = 1000  # episodes per evaluation
ALPHA1_RANGE = [20.0, 25.0, 30.0, 36.0, 40.0, 45.0, 50.0]
ALPHA2_RANGE = [12.0, 16.0, 18.0, 22.0, 26.0, 30.0, 35.0]

# === Output file ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = "search_results"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f"search_alpha_{timestamp}.log")

results = []

for a1, a2 in itertools.product(ALPHA1_RANGE, ALPHA2_RANGE):
    cmd = [
        "python", "test.py",
        "--path", PATH,
        "--epi", str(EPI),
        "--no-video",
        "--cbf-alpha1", str(a1),
        "--cbf-alpha2", str(a2),
        # "--cbf-std-alpha1", str(a1),
        # "--cbf-std-alpha", str(a2),
    ]
    print(f"\n{'='*60}")
    print(f"Testing alpha1={a1}, alpha2={a2}")
    print(f"{'='*60}")

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = out.stdout + out.stderr

        # Parse final summary line:
        # reward: X.XXX, ... safe_rate: XX.XXX%
        reward_match = re.search(r"reward: ([-\d.]+),.*safe_rate: ([\d.]+)%", output)
        last_reward_all = re.findall(r"last_reward: ([-\d.]+),", output)
        last_dist_all = re.findall(r"last_dist: ([-\d.]+),", output)

        if reward_match:
            reward = float(reward_match.group(1))
            safe_rate = float(reward_match.group(2))
            last_reward = float(last_reward_all[-1]) if last_reward_all else -1
            last_dist = float(last_dist_all[-1]) if last_dist_all else -1
            results.append((a1, a2, reward, safe_rate, last_reward, last_dist))
            print(f"  -> reward={reward:.3f}, safe_rate={safe_rate:.1f}%, last_reward={last_reward:.4f}, last_dist={last_dist:.4f}")
        else:
            print(f"  -> Failed to parse output")
            print(output[-500:])
    except subprocess.TimeoutExpired:
        print(f"  -> Timeout")
    except Exception as e:
        print(f"  -> Error: {e}")

# === Print summary ===
print(f"\n{'='*60}")
print(f"{'RESULTS SUMMARY':^60}")
print(f"{'='*60}")
print(f"{'alpha1':>8} {'alpha2':>8} {'reward':>10} {'safe_rate':>10} {'last_reward':>12} {'last_dist':>10}")
print(f"{'-'*62}")

for a1, a2, reward, sr, lr, ld in sorted(results, key=lambda x: (-x[3], -x[2])):
    marker = " ***" if sr >= 99.9 else ""
    print(f"{a1:>8.1f} {a2:>8.1f} {reward:>10.3f} {sr:>9.1f}% {lr:>12.4f} {ld:>10.4f}{marker}")

# Find best: highest safe_rate, then highest reward
if results:
    best = max(results, key=lambda x: (x[3], x[2]))
    print(f"\nBest: alpha1={best[0]}, alpha2={best[1]} "
          f"(reward={best[2]:.3f}, safe_rate={best[3]:.1f}%, last_reward={best[4]:.4f}, last_dist={best[5]:.4f})")

# === Save results to log file ===
with open(SAVE_PATH, "w") as f:
    f.write(f"Search Alpha Results - {timestamp}\n")
    f.write(f"PATH: {PATH}\n")
    f.write(f"EPI: {EPI}\n")
    f.write(f"GPU: {args.gpu}\n")
    f.write(f"ALPHA1_RANGE: {ALPHA1_RANGE}\n")
    f.write(f"ALPHA2_RANGE: {ALPHA2_RANGE}\n")
    f.write(f"\n{'alpha1':>8} {'alpha2':>8} {'reward':>10} {'safe_rate':>10} {'last_reward':>12} {'last_dist':>10}\n")
    f.write(f"{'-'*62}\n")
    for a1, a2, reward, sr, lr, ld in sorted(results, key=lambda x: (-x[3], -x[2])):
        marker = " ***" if sr >= 99.9 else ""
        f.write(f"{a1:>8.1f} {a2:>8.1f} {reward:>10.3f} {sr:>9.1f}% {lr:>12.4f} {ld:>10.4f}{marker}\n")
    if best:
        f.write(f"\nBest: alpha1={best[0]}, alpha2={best[1]} "
                f"(reward={best[2]:.3f}, safe_rate={best[3]:.1f}%, last_reward={best[4]:.4f}, last_dist={best[5]:.4f})\n")
print(f"\nResults saved to {SAVE_PATH}")
