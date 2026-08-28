#!/usr/bin/env bash
# Run the CPR task across every logprob-capable model in data/models.json,
# for one experimental condition (default value / group extraction / payoff).
# Models keyed with a leading underscore (e.g. _no_logprobs_*) are skipped,
# since their providers do not return top_logprobs and get_probs_task will raise.
#
# By default no --default and no --group_ext/--payoff are sent, i.e. the
# plain "no default, no group extraction" condition. Pass them in to run a
# different condition:
#
#   ./run_all_models.sh
#   ./run_all_models.sh --default=5
#   ./run_all_models.sh --group_ext=0 --payoff=20
#   ./run_all_models.sh --default=5 --group_ext=0 --payoff=20
#
# All output_subdir must be given as the experiment name; every condition of
# the same experiment can share one folder since main.py encodes the
# condition (default=.../group_ext=.../payoff=...) in each output filename.
#
# After all models finish, a combined plot overlaying every model's curve
# for this condition (colored by model) is saved into
# output/<output_subdir>/images/.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

TASK="cpr"
VERSION="0"
NSAMPLE="1"
BLINDING="--blinding"
OUTPUT_SUBDIR="default_exp"

# Unset by default: no default value, no group extraction.
DEFAULT=""
GROUP_EXT=""
PAYOFF=""

# Allow overrides from the command line, e.g. --group_ext=50 --payoff=10
for arg in "$@"; do
  case $arg in
    --version=*)       VERSION="${arg#*=}" ;;
    --nsample=*)       NSAMPLE="${arg#*=}" ;;
    --group_ext=*)     GROUP_EXT="${arg#*=}" ;;
    --payoff=*)        PAYOFF="${arg#*=}" ;;
    --default=*)       DEFAULT="${arg#*=}" ;;
    --output_subdir=*) OUTPUT_SUBDIR="${arg#*=}" ;;
    --no-blinding)     BLINDING="" ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ ( -n "$GROUP_EXT" && -z "$PAYOFF" ) || ( -z "$GROUP_EXT" && -n "$PAYOFF" ) ]]; then
  echo "--group_ext and --payoff must be given together" >&2
  exit 1
fi

DEFAULT_ARGS=()
if [[ -n "$DEFAULT" ]]; then
  DEFAULT_ARGS=(--default="$DEFAULT")
fi

GROUP_EXT_ARGS=()
if [[ -n "$GROUP_EXT" ]]; then
  GROUP_EXT_ARGS=(--group_ext="$GROUP_EXT" --payoff="$PAYOFF")
fi

MODELS=$(python3 -c "
import json
with open('data/models.json') as f:
    d = json.load(f)
print('\n'.join(k for k in d if not k.startswith('_')))
")

if [[ -z "$MODELS" ]]; then
  echo "No models found in data/models.json" >&2
  exit 1
fi

TOTAL=$(echo "$MODELS" | wc -l | tr -d ' ')
echo "Running $TASK across $TOTAL models"
echo "  version=$VERSION nsample=$NSAMPLE default=${DEFAULT:-none} group_ext=${GROUP_EXT:-none} payoff=${PAYOFF:-none}"
echo "  output_subdir=$OUTPUT_SUBDIR"
echo

SUCCEEDED=()
FAILED=()

i=0
while IFS= read -r model; do
  i=$((i + 1))
  echo "=============================================================="
  echo "[$i/$TOTAL] $model"
  echo "=============================================================="

  if python3 src/main.py \
      --model="$model" \
      --task="$TASK" \
      --version="$VERSION" \
      --nsample="$NSAMPLE" \
      $BLINDING \
      "${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"}" \
      "${GROUP_EXT_ARGS[@]+"${GROUP_EXT_ARGS[@]}"}" \
      --output_subdir="$OUTPUT_SUBDIR"; then
    SUCCEEDED+=("$model")
  else
    echo ">>> FAILED: $model (continuing)" >&2
    FAILED+=("$model")
  fi
  echo
done <<< "$MODELS"

echo "=============================================================="
echo "Succeeded (${#SUCCEEDED[@]}):"
for m in "${SUCCEEDED[@]+"${SUCCEEDED[@]}"}"; do echo "  $m"; done
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "Failed (${#FAILED[@]}):"
  for m in "${FAILED[@]}"; do echo "  $m"; done
fi

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
  echo
  echo "=============================================================="
  echo "Building combined (all-models) plot for this condition"
  echo "=============================================================="
  python3 src/plot_combined.py \
      --task="$TASK" \
      --version="$VERSION" \
      --nsample="$NSAMPLE" \
      $BLINDING \
      "${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"}" \
      "${GROUP_EXT_ARGS[@]+"${GROUP_EXT_ARGS[@]}"}" \
      --output_subdir="$OUTPUT_SUBDIR"
fi

echo
echo "=============================================================="
echo "Consolidating results for this condition into a CSV"
echo "=============================================================="
python3 src/consolidate.py \
    --input_dir="output/$OUTPUT_SUBDIR" \
    --task="$TASK" \
    --version="$VERSION" \
    --nsample="$NSAMPLE" \
    $BLINDING \
    "${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"}" \
    "${GROUP_EXT_ARGS[@]+"${GROUP_EXT_ARGS[@]}"}"
