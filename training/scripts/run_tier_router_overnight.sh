#!/usr/bin/env bash
set -u

TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$TRAINING_DIR/.." && pwd)"
OUTPUT_ROOT="$TRAINING_DIR/output/tier_router_v1"
RESULT_ROOT="$TRAINING_DIR/results/tier_router_v1"
PERSIST_ROOT="/teamspace/studios/this_studio/tarsiq_artifacts/tier_router_v1"
PYTHON="$TRAINING_DIR/.venv/bin/python"
BRANCH="baseline3-training"

cd "$REPO_DIR"

# Metrics for completed runs are versioned in Git, while their large checkpoints
# live outside Git. Hydrate the summaries so selection remains apples-to-apples.
for completed_name in ce ordinal_w1; do
  if [[ ! -f "$OUTPUT_ROOT/$completed_name/summary.json" && -f "$RESULT_ROOT/$completed_name/summary.json" ]]; then
    mkdir -p "$OUTPUT_ROOT/$completed_name"
    cp "$RESULT_ROOT/$completed_name/summary.json" "$OUTPUT_ROOT/$completed_name/summary.json"
    [[ -f "$RESULT_ROOT/$completed_name/history.json" ]] && cp "$RESULT_ROOT/$completed_name/history.json" "$OUTPUT_ROOT/$completed_name/history.json"
  fi
done

persist_run() {
  local name="$1"
  mkdir -p "$PERSIST_ROOT/$name" "$RESULT_ROOT/$name"
  rsync -a "$OUTPUT_ROOT/$name/" "$PERSIST_ROOT/$name/"
  for file in history.json summary.json test_metrics.json test_predictions.parquet; do
    if [[ -f "$OUTPUT_ROOT/$name/$file" ]]; then
      case "$file" in
        *.json) cp "$OUTPUT_ROOT/$name/$file" "$RESULT_ROOT/$name/$file" ;;
      esac
    fi
  done
  if [[ -f "$OUTPUT_ROOT/$name/best/run_config.json" ]]; then
    cp "$OUTPUT_ROOT/$name/best/run_config.json" "$RESULT_ROOT/$name/run_config.json"
  elif [[ -f "$OUTPUT_ROOT/$name/latest/run_config.json" ]]; then
    cp "$OUTPUT_ROOT/$name/latest/run_config.json" "$RESULT_ROOT/$name/run_config.json"
  fi
  git add "training/results/tier_router_v1/$name" 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "Persist Tier Router result: $name" || true
    git push origin "$BRANCH" || true
  fi
}

run_training() {
  local name="$1" loss="$2" weight="$3" no_struct="${4:-false}"
  if [[ -f "$OUTPUT_ROOT/$name/summary.json" ]] && grep -q '"completed": true' "$OUTPUT_ROOT/$name/summary.json"; then
    echo "SKIP_COMPLETED $name"
    persist_run "$name"
    return 0
  fi
  local args=(-m src.run_tier_router_v1 --name "$name" --loss "$loss" --under-weight "$weight")
  [[ "$no_struct" == "true" ]] && args+=(--no-struct)
  cd "$TRAINING_DIR"
  "$PYTHON" "${args[@]}" 2>&1 | tee "$OUTPUT_ROOT/${name}_console.log"
  local status=${PIPESTATUS[0]}
  cd "$REPO_DIR"
  persist_run "$name"
  if [[ $status -ne 0 ]]; then
    echo "RUN_FAILED $name status=$status"
    exit "$status"
  fi
}

run_training ordinal_w2 ordinal 2
run_training ordinal_w3 ordinal 3
run_training ordinal_w5 ordinal 5

BEST_SPEC="$($PYTHON - <<'PY'
import json
from pathlib import Path
root=Path('training/output/tier_router_v1')
specs={'ce':('ce',1.0),'ordinal_w1':('ordinal',1.0),'ordinal_w2':('ordinal',2.0),'ordinal_w3':('ordinal',3.0),'ordinal_w5':('ordinal',5.0)}
valid=[]
for name,(loss,weight) in specs.items():
    path=root/name/'summary.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data.get('completed'): valid.append((data['best_selection_score'],name,loss,weight))
if len(valid)!=len(specs): raise SystemExit('Selection stage incomplete')
_,name,loss,weight=min(valid)
print(name,loss,weight)
PY
)"
read -r BEST_NAME BEST_LOSS BEST_WEIGHT <<< "$BEST_SPEC"
NO_STRUCT_NAME="${BEST_NAME}_no_struct"
run_training "$NO_STRUCT_NAME" "$BEST_LOSS" "$BEST_WEIGHT" true

FINAL_SPEC="$($PYTHON - "$BEST_NAME" "$NO_STRUCT_NAME" <<'PY'
import json,sys
from pathlib import Path
root=Path('training/output/tier_router_v1')
names=sys.argv[1:]
best=min((json.loads((root/n/'summary.json').read_text())['best_selection_score'],n) for n in names)[1]
cfg=json.loads((root/best/'best'/'run_config.json').read_text())
print(best,cfg['loss'],cfg['under_weight'],str(not cfg['structural']).lower())
PY
)"
read -r FINAL_NAME FINAL_LOSS FINAL_WEIGHT FINAL_NO_STRUCT <<< "$FINAL_SPEC"
FINAL_CHECKPOINT="$OUTPUT_ROOT/$FINAL_NAME/best"
if [[ ! -f "$FINAL_CHECKPOINT/model.pt" ]]; then
  # A migrated Git summary can win selection even when its large checkpoint was
  # stored on the previous VM. Rebuild only that winner with the identical setup.
  REBUILD_NAME="${FINAL_NAME}_checkpoint_rebuild"
  run_training "$REBUILD_NAME" "$FINAL_LOSS" "$FINAL_WEIGHT" "$FINAL_NO_STRUCT"
  FINAL_CHECKPOINT="$OUTPUT_ROOT/$REBUILD_NAME/best"
fi
TEST_NAME="${FINAL_NAME}_strict_test"
if [[ ! -f "$OUTPUT_ROOT/$TEST_NAME/test_metrics.json" ]]; then
  cd "$TRAINING_DIR"
  args=(-m src.run_tier_router_v1 --name "$TEST_NAME" --loss "$FINAL_LOSS" --under-weight "$FINAL_WEIGHT" --eval-test --checkpoint "$FINAL_CHECKPOINT")
  [[ "$FINAL_NO_STRUCT" == "true" ]] && args+=(--no-struct)
  "$PYTHON" "${args[@]}" 2>&1 | tee "$OUTPUT_ROOT/${TEST_NAME}_console.log"
  status=${PIPESTATUS[0]}
  cd "$REPO_DIR"
  persist_run "$TEST_NAME"
  [[ $status -eq 0 ]] || exit "$status"
fi

echo "TIER_ROUTER_SEED17_COMPLETE final=$FINAL_NAME test=$TEST_NAME"
