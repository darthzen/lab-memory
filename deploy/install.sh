#!/bin/bash
# lab-memory: Karakeep install/upgrade (T-001). Drafted by qwen3-coder:30b, reviewed by Claude.
set -euo pipefail

NS=lab-memory
RELEASE=karakeep
CHART=karakeep-app/karakeep
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Idempotent secret handling. The chart regenerates random secrets on every
# upgrade if these are not pinned, which invalidates sessions and breaks the
# meilisearch key -- so we generate once and reuse.
SECRETS_FILE="$SCRIPT_DIR/secrets.env"
if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Creating $SECRETS_FILE with random secrets..."
  NEXTAUTH_SECRET=$(openssl rand -base64 36)
  MEILI_MASTER_KEY=$(openssl rand -base64 36 | tr -dc 'A-Za-z0-9')
  cat > "$SECRETS_FILE" <<EOS
NEXTAUTH_SECRET=$NEXTAUTH_SECRET
MEILI_MASTER_KEY=$MEILI_MASTER_KEY
EOS
  chmod 600 "$SECRETS_FILE"
  echo "WARNING: $SECRETS_FILE must never be committed to git (covered by .gitignore)."
fi

# shellcheck disable=SC1090
source "$SECRETS_FILE"
if [[ -z "${NEXTAUTH_SECRET:-}" ]] || [[ -z "${MEILI_MASTER_KEY:-}" ]]; then
  echo "Error: NEXTAUTH_SECRET or MEILI_MASTER_KEY is empty in $SECRETS_FILE" >&2
  exit 1
fi

helm repo add karakeep-app https://karakeep-app.github.io/helm-charts 2>/dev/null || true
helm repo update karakeep-app

kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

helm upgrade --install "$RELEASE" "$CHART" -n "$NS" -f "$SCRIPT_DIR/values.yaml" \
  --set applicationSecretKey="$NEXTAUTH_SECRET" \
  --set meilisearchMasterKey="$MEILI_MASTER_KEY" \
  --wait --timeout 10m

if ! kubectl -n "$NS" rollout status statefulset/karakeep --timeout=300s; then
  echo "Statefulset rollout not confirmed under that name; listing pods..."
  kubectl -n "$NS" get pods
fi

echo "=== Status ==="
kubectl -n "$NS" get pods,pvc,ingress

echo "=== Next steps ==="
echo "1. Create the first user in the web UI (http://karakeep.ash4d.com)"
echo "2. Then set DISABLE_SIGNUPS=true in values.yaml and re-run this script"
