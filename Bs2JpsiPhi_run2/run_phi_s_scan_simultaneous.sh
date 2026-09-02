#!/bin/bash
# =====================================================================
# Batch SIMULTANEOUS pseudo_data workflow over 50 randomly-sampled seeds.
#
# Chain per seed (all _simultaneous variants):
#   1. create_pseudo_data_simultaneous.py --seed S  (per year×trigger split)
#   2. cut_pseudo_data_simultaneous.py    --seed S  (per sub-sample time>0)
#   3. double_pseudo_data_simultaneous.py            (per sub-sample CP double)
#   4. fit_conv_pseudo_data_simultaneous.py          (MultiConfig simultaneous fit)
#
# Only the FIT log is saved per seed (phi_s_scan_logs_simult/run_seed{S}.log).
#
# After each fit, results are read from:
#   1. final_params_conv_pseudo_data_simultaneous_pseudo_data_{suffix}.json
#      raw parameters listed in PARAMS (keys: value / error)
#   2. trans_params_conv_{suffix}.json
#      polarization quantities (per-key dicts with value/error)
# n_sigma = |value - ref| / error, appended to phi_s_scan_results_simultaneous.txt
# Special case: if Bs_delta_gamma is in PARAMS, written value is -Bs_delta_gamma.
# NOTE: results file cleared at the start of each run.
# =====================================================================
set -u
cd "$(dirname "$0")"

# ------------------------- configuration ---------------------------
NSEEDS=50
SEED_MIN=0
SEED_MAX=500
MASTER_SEED=2026
YEARS="2015,2016,2017,2018"
TRIGGER="all"
RESULTS="phi_s_scan_results_simultaneous.txt"
SEEDLIST="phi_s_scan_seeds_simultaneous.txt"
LOGDIR="phi_s_scan_logs_simultaneous"

PARAMS=("Bs_poqi" "Bs_delta_gamma" "Bs_gamma")
REFS=("-0.03" "0.08543" "0.6614")

POLAR_PARAMS=("|A0|^2" "|A∥|^2" "|A⊥|^2" "δ∥ - δ0" "δ⊥ - δ0")
POLAR_REFS=("0.524176" "0.225625" "0.250000" "3.26" "3.08")
CHECK_POLAR=false
# --------------------------------------------------------------------

mkdir -p "${LOGDIR}"
years_suffix=$(echo "${YEARS}" | tr ',' '_')
file_suffix="simultaneous_${years_suffix}_${TRIGGER}"
json_file="final_params_conv_pseudo_data_simultaneous_pseudo_data_${file_suffix}.json"
trans_json="trans_params_conv_${file_suffix}.json"

# ---- header / (name:ref) pairs --------------------------------------
HEADER="seed"
PAIRS=()
for j in "${!PARAMS[@]}"; do
    HEADER+=$'\t'"${PARAMS[$j]}_value"$'\t'"${PARAMS[$j]}_error"$'\t'"${PARAMS[$j]}_reference"$'\t'"${PARAMS[$j]}_n_sigma"
    PAIRS+=("${PARAMS[$j]}:${REFS[$j]}")
done
POLAR_PAIRS=()
if ${CHECK_POLAR}; then
    for j in "${!POLAR_PARAMS[@]}"; do
        HEADER+=$'\t'"${POLAR_PARAMS[$j]}_value"$'\t'"${POLAR_PARAMS[$j]}_error"$'\t'"${POLAR_PARAMS[$j]}_reference"$'\t'"${POLAR_PARAMS[$j]}_n_sigma"
        POLAR_PAIRS+=("${POLAR_PARAMS[$j]}:${POLAR_REFS[$j]}")
    done
fi
# clear the results file once per run (before the seed loop)
printf "%s\n" "${HEADER}" > "${RESULTS}"

# ---- 1. sample 50 unique seeds from [SEED_MIN, SEED_MAX] -----------
mapfile -t SEEDS < <(python3 - "${MASTER_SEED}" "${NSEEDS}" "${SEED_MIN}" "${SEED_MAX}" <<'PYEOF'
import random, sys
master, n, lo, hi = map(int, sys.argv[1:5])
random.seed(master)
print("\n".join(map(str, random.sample(range(lo, hi + 1), n))))
PYEOF
)
printf "%s\n" "${SEEDS[@]}" > "${SEEDLIST}"
echo "Sampled ${#SEEDS[@]} seeds (master_seed=${MASTER_SEED}): ${SEEDS[*]}"
echo "Seed list saved to ${SEEDLIST}"

# ---- 2. loop over seeds --------------------------------------------
i=0
for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    log="${LOGDIR}/run_seed${seed}.log"
    echo ""
    echo "=========================================================="
    echo "[${i}/${#SEEDS[@]}] seed = ${seed}   (fit log: ${log})"
    echo "=========================================================="

    # ---- 2a-2c. create / cut / double (output discarded) ----
    python3 create_pseudo_data_simultaneous.py --seed "${seed}" --years "${YEARS}" --trigger "${TRIGGER}" > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then echo "FAILED at create_pseudo_data_simultaneous (seed ${seed})"; continue; fi

    python3 cut_pseudo_data_simultaneous.py --seed "${seed}" --years "${YEARS}" --trigger "${TRIGGER}" > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then echo "FAILED at cut_pseudo_data_simultaneous (seed ${seed})"; continue; fi

    python3 double_pseudo_data_simultaneous.py --years "${YEARS}" --trigger "${TRIGGER}" > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then echo "FAILED at double_pseudo_data_simultaneous (seed ${seed})"; continue; fi

    # ---- 2d. simultaneous fit (only this log is kept) ----
    python3 fit_conv_pseudo_data_simultaneous.py --years "${YEARS}" --trigger "${TRIGGER}" \
        > "${log}" 2>&1
    if [[ $? -ne 0 ]]; then echo "FAILED at fit_conv_pseudo_data_simultaneous (seed ${seed}), see ${log}"; continue; fi

    # ---- 2e. extract listed parameters, compute n_sigma, append ----
    line=$(python3 - "${json_file}" "${trans_json}" "${PAIRS[@]}" "--" "${POLAR_PAIRS[@]}" <<'PYEOF'
import json, sys
final_path, trans_path = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
sep = rest.index("--")
raw_pairs, polar_pairs = rest[:sep], rest[sep + 1:]
with open(final_path) as f:
    final = json.load(f)
trans = {}
if polar_pairs:
    with open(trans_path) as f:
        trans = json.load(f)

cols = []
# raw parameters: final JSON has flat value / error dicts
for pair in raw_pairs:
    name, ref = pair.rsplit(":", 1)
    ref = float(ref)
    v = final["value"].get(name)
    e = final["error"].get(name, 0.0)
    if name == "Bs_delta_gamma" and v is not None:
        v = -v  # compare/write -Bs_delta_gamma, not the parameter itself
    if v is None:
        cols.extend(["nan", "nan", f"{ref:.6g}", "nan"])
        continue
    ns = abs(v - ref) / e if e > 0 else float("inf")
    cols.extend([f"{v:.6f}", f"{e:.6f}", f"{ref:.6g}", f"{ns:.3f}"])
# polarization quantities: trans JSON has per-key dicts with value/error
for pair in polar_pairs:
    name, ref = pair.rsplit(":", 1)
    ref = float(ref)
    entry = trans.get(name, {})
    v = entry.get("value")
    e = entry.get("error", 0.0)
    if v is None:
        cols.extend(["nan", "nan", f"{ref:.6g}", "nan"])
        continue
    ns = abs(v - ref) / e if e > 0 else float("inf")
    cols.extend([f"{v:.6f}", f"{e:.6f}", f"{ref:.6g}", f"{ns:.3f}"])
print("\t".join(cols))
PYEOF
)
    if [[ -n "${line}" ]]; then
        printf "%d\t%s\n" "${seed}" "${line}" >> "${RESULTS}"
        echo "seed=${seed}  ->  ${line}"
    else
        echo "WARNING: could not parse ${json_file} for seed ${seed}, see ${log}"
    fi
done

echo ""
echo "=========================================================="
echo "Done. Results in ${RESULTS}, fit logs in ${LOGDIR}/"
echo "=========================================================="
