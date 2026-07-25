#!/bin/bash
# Sweep test: vary num_agents and obs across seeds, collect results.
# Usage:
#   ./sweep_test.sh <model_path_or_parent> [episodes]
#
# If <model_path_or_parent> contains seed* subdirs (e.g. logs/LidarSpread/informarl_subgoal),
# all seeds are run and aggregated (mean ± std). Otherwise treated as a single seed dir.

INPUT_PATH="${1:-logs/LidarBicycleTarget/informarl_subgoal}"
EPI="${2:-1000}"

# Bicycle is harder to settle within the default 0.01 dist2goal — relax to 0.05.
EXTRA_ARGS=()
if [[ "$INPUT_PATH" == *Bicycle* ]]; then
    EXTRA_ARGS+=(--reach-thresh 0.1)
fi

# Result filename derived from input path: logs/LidarSpread/informarl_subgoal -> LidarSpread_informarl_subgoal
LABEL=$(echo "$INPUT_PATH" | sed 's|/$||' | sed 's|^logs/||' | sed 's|/|_|g')
RESULT_FILE="results_sweep/sweep_results_${LABEL}.txt"

# Detect seeds: if INPUT_PATH has seed* subdirs, run all of them.
SEED_DIRS=()
if compgen -G "$INPUT_PATH/seed*" > /dev/null; then
    for d in "$INPUT_PATH"/seed*/; do
        SEED_DIRS+=("${d%/}")
    done
else
    SEED_DIRS+=("$INPUT_PATH")
fi

RAW_FILE=$(mktemp)
trap "rm -f $RAW_FILE" EXIT

mkdir -p "$(dirname "$RESULT_FILE")"

echo "=== Sweep Test ===" | tee "$RESULT_FILE"
echo "Input path: $INPUT_PATH" | tee -a "$RESULT_FILE"
echo "Seeds found: ${#SEED_DIRS[@]}" | tee -a "$RESULT_FILE"
for sd in "${SEED_DIRS[@]}"; do echo "  - $sd" | tee -a "$RESULT_FILE"; done
echo "Episodes per config: $EPI" | tee -a "$RESULT_FILE"
echo "Started at: $(date)" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

# === Reward coefficients (from dgppo/trainer/utils.py) ===
echo "--- Reward coefficients (dgppo/trainer/utils.py) ---" | tee -a "$RESULT_FILE"
grep -E "^(GOAL_REWARD_COEF|SUBGOAL_BONUS_THRESH|SUBGOAL_BONUS_COEF|DIST_TO_GOAL_COEF|SUBGOAL_SHADOW_COEF)\s*=" \
    dgppo/trainer/utils.py | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

printf "%-12s %-6s %-6s %-22s %-12s %-12s %-14s %-16s %-16s\n" \
    "sweep" "n" "obs" "seed" "last_dist" "safe_rate" "reach_agent" "success_agent" "success_epi" \
    | tee -a "$RESULT_FILE"
printf '%140s\n' '' | tr ' ' '=' | tee -a "$RESULT_FILE"

run_test() {
    local sweep_name="$1"
    local n_agents="$2"
    local n_obs="$3"
    local model_path="$4"
    local seed_label="$5"

    echo ">>> [$seed_label] $sweep_name | n=$n_agents, obs=$n_obs"

    output=$(python test_manifold.py --path "$model_path" -n "$n_agents" --obs "$n_obs" \
        --epi "$EPI" --no-video "${EXTRA_ARGS[@]}" 2>&1)

    summary=$(echo "$output" | grep "^reward:")

    if [ -z "$summary" ]; then
        printf "%-12s %-6s %-6s %-22s FAILED\n" \
            "$sweep_name" "$n_agents" "$n_obs" "$seed_label" | tee -a "$RESULT_FILE"
        return
    fi

    last_dist=$(echo "$summary"   | grep -oP 'last_dist: \K[-0-9.]+')
    safe_rate=$(echo "$summary"   | grep -oP 'safe_rate: \K[-0-9.]+')
    reach_agent=$(echo "$summary" | grep -oP 'reach_agent: \K[-0-9.]+')
    success_agent=$(echo "$summary" | grep -oP 'success_agent: \K[-0-9.]+')
    success_epi=$(echo "$summary" | grep -oP 'success_epi: \K[-0-9.]+')

    printf "%-12s %-6s %-6s %-22s %-12s %-12s %-14s %-16s %-16s\n" \
        "$sweep_name" "$n_agents" "$n_obs" "$seed_label" \
        "$last_dist" "${safe_rate}%" "${reach_agent}%" "${success_agent}%" "${success_epi}%" \
        | tee -a "$RESULT_FILE"

    # Raw record for aggregation: sweep n obs seed last_dist safe reach_agent success_agent success_epi
    echo "$sweep_name $n_agents $n_obs $seed_label $last_dist $safe_rate $reach_agent $success_agent $success_epi" \
        >> "$RAW_FILE"
}

for seed_path in "${SEED_DIRS[@]}"; do
    seed_label=$(basename "$seed_path")

    echo "" | tee -a "$RESULT_FILE"
    echo "########## Seed: $seed_label ##########" | tee -a "$RESULT_FILE"

    # === Sweep 1: fixed obs=3, vary num_agents ===
    echo "--- Sweep 1: fixed obs=3, vary num_agents ---" | tee -a "$RESULT_FILE"
    for n in 3 6 9 12 15 18 21; do
        run_test "vary_n" "$n" 3 "$seed_path" "$seed_label"
    done

    # === Sweep 2: fixed num_agents=3, vary obs ===
    echo "--- Sweep 2: fixed num_agents=3, vary obs ---" | tee -a "$RESULT_FILE"
    for obs in 3 6 9 12 15 18 21; do
        run_test "vary_obs" 3 "$obs" "$seed_path" "$seed_label"
    done
done

# === Aggregation: mean ± std across seeds per (sweep, n, obs) ===
echo "" | tee -a "$RESULT_FILE"
echo "=== Aggregated (mean ± std across ${#SEED_DIRS[@]} seed(s)) ===" | tee -a "$RESULT_FILE"
printf "%-12s %-6s %-6s %-18s %-18s %-18s %-18s %-18s %-6s\n" \
    "sweep" "n" "obs" "last_dist" "safe_rate" "reach_agent" "success_agent" "success_epi" "n_seed" \
    | tee -a "$RESULT_FILE"
printf '%140s\n' '' | tr ' ' '=' | tee -a "$RESULT_FILE"

awk '
function stat(sum, sumsq, n,    mean, var) {
    mean = sum / n
    if (n > 1) {
        var = (sumsq - n * mean * mean) / (n - 1)
        if (var < 0) var = 0
        return mean "|" sqrt(var)
    }
    return mean "|0"
}
{
    key = $1 "\t" $2 "\t" $3
    cnt[key]++
    sum_d[key]  += $5; sq_d[key]  += $5 * $5
    sum_s[key]  += $6; sq_s[key]  += $6 * $6
    sum_r[key]  += $7; sq_r[key]  += $7 * $7
    sum_sa[key] += $8; sq_sa[key] += $8 * $8
    sum_se[key] += $9; sq_se[key] += $9 * $9
    order[key] = ($1 == "vary_n" ? 0 : 1) * 1000000 + $2 * 1000 + $3
}
END {
    i = 0
    for (k in cnt) { keys[i++] = k }
    n_keys = i
    for (a = 0; a < n_keys - 1; a++) {
        for (b = a + 1; b < n_keys; b++) {
            if (order[keys[a]] > order[keys[b]]) {
                tmp = keys[a]; keys[a] = keys[b]; keys[b] = tmp
            }
        }
    }
    for (a = 0; a < n_keys; a++) {
        k = keys[a]; n = cnt[k]
        d  = stat(sum_d[k],  sq_d[k],  n)
        s  = stat(sum_s[k],  sq_s[k],  n)
        r  = stat(sum_r[k],  sq_r[k],  n)
        sa = stat(sum_sa[k], sq_sa[k], n)
        se = stat(sum_se[k], sq_se[k], n)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\n", k, d, s, r, sa, se, n
    }
}' "$RAW_FILE" | while IFS=$'\t' read sw n o d s r sa se ns; do
    d_mean=${d%|*};   d_std=${d#*|}
    s_mean=${s%|*};   s_std=${s#*|}
    r_mean=${r%|*};   r_std=${r#*|}
    sa_mean=${sa%|*}; sa_std=${sa#*|}
    se_mean=${se%|*}; se_std=${se#*|}
    d_str=$(printf  "%.4f±%.4f"   "$d_mean"  "$d_std")
    s_str=$(printf  "%.2f±%.2f%%" "$s_mean"  "$s_std")
    r_str=$(printf  "%.2f±%.2f%%" "$r_mean"  "$r_std")
    sa_str=$(printf "%.2f±%.2f%%" "$sa_mean" "$sa_std")
    se_str=$(printf "%.2f±%.2f%%" "$se_mean" "$se_std")
    printf "%-12s %-6s %-6s %-18s %-18s %-18s %-18s %-18s %-6s\n" \
        "$sw" "$n" "$o" "$d_str" "$s_str" "$r_str" "$sa_str" "$se_str" "$ns" \
        | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "Finished at: $(date)" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
