mkdir -p logs

D=12
max_jobs=30
G="ancestral"
F="nonlinear_confounder"

a_vals=(0 1 2 5 8 9 10 14 15 17)

for A in "${a_vals[@]}"; do
  for SEED in {0..9}; do
    LOG_FILE="logs/d${D}_${G}_a${A}_s${SEED}_f${F}.txt"

    echo ">>> d=${D}, a=${A}, seed=${SEED}"
    echo "    log: ${LOG_FILE}"

    python src/DagmaDCE/run_experiments.py \
      -d ${D} \
      -g ${G} \
      -s ${SEED} \
      -a ${A} \
      -f ${F} \
      > "${LOG_FILE}" 2>&1 &

    # 控制最多并行 4 个
    while [ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]; do
      sleep 3
    done
  done
done

wait
echo ">>> ✅ all jobs finished for d=${D}, a in (2 5 6 8 9 10 15 17 19 20), seeds 0–9"



pkill -f run_experiments.py

python src/DagmaDCE/run_experiments.py -realData y > logs/realData.log 2>&1