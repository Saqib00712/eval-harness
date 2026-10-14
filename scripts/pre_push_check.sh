#!/usr/bin/env bash
# Day 21: optional local pre-push check -- runs the exact same three steps
# as .github/workflows/ci.yml's free `test` job, so a failure shows up on
# your own machine before a push, not after. The same judgment call
# production-rag's own Day 11 made for its pre-push hook: optional, never
# installed automatically, because a hook that silently blocks `git push`
# without the person choosing it is a worse surprise than a red CI run.
#
# Install it (optional):
#   cp scripts/pre_push_check.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
#
# Or just run it by hand whenever:
#   bash scripts/pre_push_check.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Running test suite =="
python3 -m pytest -q

echo "== Checking coverage threshold =="
python3 -m coverage run -m pytest -q >/dev/null
python3 -m coverage report \
  --include="eval_harness/*" \
  --omit="eval_harness/providers/openai_embedding.py,eval_harness/providers/openai_llm.py" \
  --fail-under=98

echo "== Running end-to-end demo script =="
python3 scripts/demo_end_to_end.py >/dev/null

echo "All pre-push checks passed."
