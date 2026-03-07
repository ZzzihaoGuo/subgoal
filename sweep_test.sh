#!/bin/bash
# Sweep test: vary num_agents and obs, collect results

PATH_MODEL="${1:-logs/LidarSpread/informarl_subgoal/seed0_212001549_UBBT}"
EPI="${2:-1000}"
RESULT_FILE="sweep_results.txt"

echo "=== Sweep Test ===" | tee "$RESULT_FILE"
echo "Model path: $PATH_MODEL" | tee -a "$RESULT_FILE"
echo "Episodes per config: $EPI" | tee -a "$RESULT_FILE"
echo "Started at: $(date)" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

printf "%-12s %-6s %-6s %-12s %-12s %-12s %-12s %-16s %-16s\n" \
    "sweep" "n" "obs" "reward" "cost" "last_dist" "safe_rate" "success_agent" "success_epi" \
    | tee -a "$RESULT_FILE"
printf "%s\n" "$(printf '=%.0s' {1..120})" | tee -a "$RESULT_FILE"

run_test() {
    local sweep_name="$1"
    local n_agents="$2"
    local n_obs="$3"

    echo ">>> Running: $sweep_name | n=$n_agents, obs=$n_obs"

    output=$(python test_manifold.py --path "$PATH_MODEL" -n "$n_agents" --obs "$n_obs" \
        --epi "$EPI" --no-video 2>&1)

    summary=$(echo "$output" | grep "^reward:")

    if [ -z "$summary" ]; then
        printf "%-12s %-6s %-6s FAILED\n" "$sweep_name" "$n_agents" "$n_obs" | tee -a "$RESULT_FILE"
        return
    fi

    reward=$(echo "$summary" | grep -oP 'reward: \K[-0-9.]+')
    cost=$(echo "$summary" | grep -oP 'cost: \K[-0-9.]+')
    last_dist=$(echo "$summary" | grep -oP 'last_dist: \K[-0-9.]+')
    safe_rate=$(echo "$summary" | grep -oP 'safe_rate: \K[-0-9.]+')
    success_agent=$(echo "$summary" | grep -oP 'success_agent: \K[-0-9.]+')
    success_epi=$(echo "$summary" | grep -oP 'success_epi: \K[-0-9.]+')

    printf "%-12s %-6s %-6s %-12s %-12s %-12s %-12s %-16s %-16s\n" \
        "$sweep_name" "$n_agents" "$n_obs" \
        "$reward" "$cost" "$last_dist" \
        "${safe_rate}%" "${success_agent}%" "${success_epi}%" \
        | tee -a "$RESULT_FILE"
}

# === Sweep 1: fixed obs=3, vary num_agents ===
echo "" | tee -a "$RESULT_FILE"
echo "--- Sweep 1: fixed obs=3, vary num_agents ---" | tee -a "$RESULT_FILE"
for n in 3 5 7 9 11 15 21; do
    run_test "vary_n" "$n" 3
done

# === Sweep 2: fixed num_agents=3, vary obs ===
echo "" | tee -a "$RESULT_FILE"
echo "--- Sweep 2: fixed num_agents=3, vary obs ---" | tee -a "$RESULT_FILE"
for obs in 5 7 9 11 15; do
    run_test "vary_obs" 3 "$obs"
done

echo "" | tee -a "$RESULT_FILE"
echo "Finished at: $(date)" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE"
