#!/bin/bash
#SBATCH --job-name=explorer_app
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=/home/aih/dinesh.haridoss/logs/explorer_app_%j.out

# ── Auto-resubmit on wall-time (24/7 operation) ───────────────────────────────
SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'Command=\K\S+' || echo "/ictstr01/home/aih/dinesh.haridoss/chicago_mil/patient_explorer/submit_app.sh")"
trap 'echo "[wall-time] Resubmitting for 24/7 operation..."; sbatch '"$SCRIPT_PATH"'; kill $CF_PID 2>/dev/null; exit 0' USR1

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /ictstr01/home/aih/dinesh.haridoss/chicago_mil/patient_explorer

echo "Running on: $(hostname)"

# ── Reverse SSH tunnel (for on-campus / VPN access) ────────────────────────
ssh -o StrictHostKeyChecking=no \
    -o IdentitiesOnly=yes \
    -o ExitOnForwardFailure=no \
    -i /home/aih/dinesh.haridoss/.ssh/hpc_cluster.key \
    -fNR 8501:localhost:8501 hpc-submit01.scidom.de && \
    echo "SSH tunnel OK → hpc-submit01:8501" || \
    echo "SSH tunnel failed (Cloudflare will still work)"

# ── Cloudflare quick tunnel (public HTTPS, URL changes per job) ───────────────
CFDIR=/home/aih/dinesh.haridoss/.local/bin
CF_LOG=/home/aih/dinesh.haridoss/logs/cloudflared_$$.log
"$CFDIR/cloudflared" tunnel --url http://localhost:8501 \
    --no-autoupdate 2>"$CF_LOG" &
CF_PID=$!

for i in $(seq 1 20); do
    CF_URL=$(grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
    [ -n "$CF_URL" ] && break
    sleep 1
done

# Write current URL to a stable file collaborators can check
URL_FILE=/home/aih/dinesh.haridoss/logs/current_app_url.txt
echo "$CF_URL" > "$URL_FILE"

# ── Update GitHub Pages redirect (permanent link auto-forwards here) ──────────
GH_TOKEN=$(cat /home/aih/dinesh.haridoss/.secrets/github_token 2>/dev/null || echo "")
GH_REPO="Dinesh-Adhithya-H/lungmil-results"
GH_FILE="index.html"

HTML="<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>LungMIL Explorer</title><script>window.location.replace(\"${CF_URL}\");</script><style>body{font-family:sans-serif;background:#0d1117;color:#c9d1d9;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}a{color:#58a6ff;font-size:1.2em}</style></head><body><div><p>Redirecting to LungMIL Explorer...</p><p><a href=\"${CF_URL}\">${CF_URL}</a></p><p style=\"font-size:.8em;color:#8b949e\">Password: lungmil2024</p></div></body></html>"
SHA=$(curl -s -H "Authorization: token $GH_TOKEN" "https://api.github.com/repos/${GH_REPO}/contents/${GH_FILE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sha',''))" 2>/dev/null)
ENCODED=$(printf '%s' "$HTML" | base64 -w 0)
curl -s -X PUT -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/${GH_REPO}/contents/${GH_FILE}" -d "{\"message\":\"update redirect\",\"content\":\"${ENCODED}\",\"sha\":\"${SHA}\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('GitHub Pages updated ✓' if 'content' in d else 'Pages update failed: '+d.get('message','?'))" 2>/dev/null || echo "GitHub Pages update failed"

echo ""
echo "════════════════════════════════════════"
echo "  PERMANENT  : https://dinesh-adhithya-h.github.io/lungmil-results/"
echo "  DIRECT URL : $CF_URL"
echo "  Password   : ${EXPLORER_PASSWORD:-lungmil2024}"
echo ""
echo "  On-campus / VPN:"
echo "  ssh -L 8501:localhost:8501 dinesh.haridoss@hpc-submit01.scidom.de"
echo "  → http://localhost:8501"
echo "════════════════════════════════════════"

export EXPLORER_DATA="$(pwd)/data"
export EXPLORER_PASSWORD="${EXPLORER_PASSWORD:-lungmil2024}"

streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 127.0.0.1 \
    2>&1

kill $CF_PID 2>/dev/null
