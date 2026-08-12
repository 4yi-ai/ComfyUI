#!/bin/sh
# 4yi App Platform entrypoint for ComfyUI.
# Runs after the persistent /data volume is mounted, so it (re)creates the
# subdirs the volume masks, seeds first-run defaults, then execs ComfyUI.
set -e

mkdir -p /data/input /data/output /data/temp /data/user/default/workflows

# Default the UI language to Chinese on a fresh install. Only written when the
# user has no settings yet, so it never clobbers a choice the user later makes.
SETTINGS=/data/user/default/comfy.settings.json
if [ ! -f "$SETTINGS" ]; then
  printf '{"Comfy.Locale": "zh"}\n' > "$SETTINGS"
fi

# Refresh the bundled 4yi example workflow(s) on every start, so existing
# installs (whose volume already has an older copy) also pick up updated
# templates on their next restart — not just brand-new installs. Only the
# bundled filenames are overwritten; the user's own saved workflows are never
# touched. To customize an official example, "Save As" a new name — an in-place
# edit to a bundled file is replaced on the next restart.
for src in /app/4yi_examples/*.json; do
  [ -e "$src" ] || continue
  cp -f "$src" "/data/user/default/workflows/$(basename "$src")"
done

# --disable-api-nodes drops ComfyUI's ~125 built-in "partner" nodes, which call
#   Comfy's own paid cloud (api.comfy.org), not the 4yi gateway, and would only
#   confuse/charge users here. Generation goes through the bundled 4yi_gateway
#   nodes instead.
# --enable-cors-header replaces the origin-only middleware that 403s cross-site
#   requests (the platform and app live on different domains; see Dockerfile).
exec python main.py \
  --listen 0.0.0.0 --port 8188 --cpu \
  --enable-cors-header '*' \
  --disable-api-nodes \
  --database-url sqlite:////data/user/comfyui.db \
  --input-directory /data/input \
  --output-directory /data/output \
  --temp-directory /data/temp \
  --user-directory /data/user
