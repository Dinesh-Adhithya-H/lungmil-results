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

echo ""
echo "════════════════════════════════════════"
echo "  PUBLIC URL : $CF_URL"
echo "  Password   : ${EXPLORER_PASSWORD:-lungmil2024}"
echo "  (URL changes on restart — always find current URL at: $URL_FILE)"
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
