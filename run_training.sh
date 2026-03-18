#!/usr/bin/env bash
# =============================================================================
# run_training.sh  –  Convenience launcher for the ME 295A vibration-control
#                     RL training pipeline.
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh [quick | full | eval | demo | csv | board]
#
#   quick  – 200 k timesteps, 2 envs   (development / smoke-test)
#   full   – 1 M timesteps,  4 envs    (paper-quality results)
#   eval   – evaluate the most recent best_model checkpoint
#   demo   – headless animation of the best model (no rendering flag)
#   csv    – regenerate vibration_training_data.csv
#   board  – launch TensorBoard on port 6006
#   (none) – defaults to quick
# =============================================================================

set -euo pipefail

MODE="${1:-quick}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  AI-Assisted Vibration Control  |  ME 295A  |  SJSU"
echo "  Mode: $MODE"
echo "============================================================"

case "$MODE" in

  quick)
    echo "[INFO] Quick training: 200 k timesteps, 2 parallel envs"
    python controller_reinforcement_agent.py train \
        --timesteps 200000 \
        --n-envs 2
    ;;

  full)
    echo "[INFO] Full training: 1 000 000 timesteps, 4 parallel envs"
    python controller_reinforcement_agent.py train \
        --timesteps 1000000 \
        --n-envs 4
    ;;

  eval)
    # Find the most recently modified best_model zip
    BEST=$(find models/ -name "best_model.zip" -printf "%T@ %p\n" 2>/dev/null \
           | sort -n | tail -1 | awk '{print $2}' | sed 's/\.zip$//')
    if [[ -z "$BEST" ]]; then
        echo "[ERROR] No best_model.zip found under models/. Run training first."
        exit 1
    fi
    echo "[INFO] Evaluating model: $BEST"
    python controller_reinforcement_agent.py eval \
        --model "$BEST" \
        --episodes 20
    ;;

  demo)
    BEST=$(find models/ -name "best_model.zip" -printf "%T@ %p\n" 2>/dev/null \
           | sort -n | tail -1 | awk '{print $2}' | sed 's/\.zip$//')
    if [[ -z "$BEST" ]]; then
        echo "[ERROR] No best_model.zip found. Run training first."
        exit 1
    fi
    echo "[INFO] Demo animation: $BEST"
    python controller_reinforcement_agent.py demo --model "$BEST"
    ;;

  csv)
    echo "[INFO] Regenerating vibration_training_data.csv ..."
    python generate_csv.py
    ;;

  board)
    echo "[INFO] Launching TensorBoard at http://localhost:6006"
    tensorboard --logdir logs/ --port 6006
    ;;

  *)
    echo "[ERROR] Unknown mode '$MODE'. Use: quick | full | eval | demo | csv | board"
    exit 1
    ;;
esac

echo "============================================================"
echo "  Done."
echo "============================================================"
