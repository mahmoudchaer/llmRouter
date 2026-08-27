#!/usr/bin/env bash
set -u

TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$TRAINING_DIR/.." && pwd)"
OUTPUT_ROOT="$TRAINING_DIR/output/tier_router_v2_3tier"
RESULT_ROOT="$TRAINING_DIR/results/tier_router_v2_3tier"
PERSIST_ROOT="/teamspace/studios/this_studio/tarsiq_artifacts/tier_router_v2_3tier"
PYTHON="$TRAINING_DIR/.venv/bin/python"
CONFIG="configs/tier_router_v2_3tier.yaml"

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT" "$PERSIST_ROOT"

persist_run() {
  local name="$1"
  mkdir -p "$PERSIST_ROOT/$name" "$RESULT_ROOT/$name"
  cp -a "$OUTPUT_ROOT/$name/." "$PERSIST_ROOT/$name/"
  for file in history.json summary.json test_metrics.json; do
    [[ -f "$OUTPUT_ROOT/$name/$file" ]] && cp "$OUTPUT_ROOT/$name/$file" "$RESULT_ROOT/$name/$file"
  done
  if [[ -f "$OUTPUT_ROOT/$name/best/run_config.json" ]]; then
    cp "$OUTPUT_ROOT/$name/best/run_config.json" "$RESULT_ROOT/$name/run_config.json"
  elif [[ -f "$OUTPUT_ROOT/$name/latest/run_config.json" ]]; then
    cp "$OUTPUT_ROOT/$name/latest/run_config.json" "$RESULT_ROOT/$name/run_config.json"
  fi
  cd "$REPO_DIR"
  git add -f "training/results/tier_router_v2_3tier/$name" 2>/dev/null || true
  git diff --cached --quiet || git commit -m "Persist 3-tier result: $name" || true
}

run_training() {
  local name="$1" loss="$2" weight="$3" no_struct="${4:-false}"
  if [[ -f "$OUTPUT_ROOT/$name/summary.json" ]] && grep -q '"completed": true' "$OUTPUT_ROOT/$name/summary.json"; then
    echo "SKIP_COMPLETED $name"; persist_run "$name"; return 0
  fi
  local args=(-m src.run_tier_router_v2 --name "$name" --loss "$loss" --under-weight "$weight" --config "$CONFIG")
  [[ "$no_struct" == "true" ]] && args+=(--no-struct)
  cd "$TRAINING_DIR"
  "$PYTHON" "${args[@]}" 2>&1 | tee "$OUTPUT_ROOT/${name}_console.log"
  local status=${PIPESTATUS[0]}
  cd "$REPO_DIR"; persist_run "$name"
  [[ $status -eq 0 ]] || exit "$status"
}

run_training ce ce 1
run_training ordinal_w1 ordinal 1
run_training ordinal_w2 ordinal 2
run_training ordinal_w3 ordinal 3
run_training ordinal_w5 ordinal 5

BEST_SPEC="$(cd "$REPO_DIR" && "$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('training/output/tier_router_v2_3tier')
specs={'ce':('ce',1.0),'ordinal_w1':('ordinal',1.0),'ordinal_w2':('ordinal',2.0),'ordinal_w3':('ordinal',3.0),'ordinal_w5':('ordinal',5.0)}
valid=[]
for name,(loss,weight) in specs.items():
    data=json.loads((root/name/'summary.json').read_text())
    valid.append((data['best_selection_score'],name,loss,weight))
_,name,loss,weight=min(valid)
print(name,loss,weight)
PY
)"
read -r BEST_NAME BEST_LOSS BEST_WEIGHT <<< "$BEST_SPEC"
NO_STRUCT_NAME="${BEST_NAME}_no_struct"
run_training "$NO_STRUCT_NAME" "$BEST_LOSS" "$BEST_WEIGHT" true

FINAL_NAME="$(cd "$REPO_DIR" && "$PYTHON" - "$BEST_NAME" "$NO_STRUCT_NAME" <<'PY'
import json,sys
from pathlib import Path
root=Path('training/output/tier_router_v2_3tier')
print(min((json.loads((root/n/'summary.json').read_text())['best_selection_score'],n) for n in sys.argv[1:])[1])
PY
)"
FINAL_CFG="$OUTPUT_ROOT/$FINAL_NAME/best/run_config.json"
FINAL_LOSS="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["loss"])' "$FINAL_CFG")"
FINAL_WEIGHT="$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["under_weight"])' "$FINAL_CFG")"
FINAL_STRUCT="$("$PYTHON" -c 'import json,sys;print(str(json.load(open(sys.argv[1]))["structural"]).lower())' "$FINAL_CFG")"
TEST_NAME="${FINAL_NAME}_strict_test"
cd "$TRAINING_DIR"
args=(-m src.run_tier_router_v2 --name "$TEST_NAME" --loss "$FINAL_LOSS" --under-weight "$FINAL_WEIGHT" --config "$CONFIG" --eval-test --checkpoint "$OUTPUT_ROOT/$FINAL_NAME/best")
[[ "$FINAL_STRUCT" == "false" ]] && args+=(--no-struct)
"$PYTHON" "${args[@]}" 2>&1 | tee "$OUTPUT_ROOT/${TEST_NAME}_console.log"
status=${PIPESTATUS[0]}; cd "$REPO_DIR"; persist_run "$TEST_NAME"
[[ $status -eq 0 ]] || exit "$status"
echo "TIER_ROUTER_V2_SEED17_COMPLETE final=$FINAL_NAME test=$TEST_NAME"
