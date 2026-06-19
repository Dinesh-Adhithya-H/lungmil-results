#!/bin/bash
#SBATCH --job-name=explorer_app
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aih/dinesh.haridoss/logs/explorer_app_%j.out

source /home/aih/dinesh.haridoss/miniconda3/etc/profile.d/conda.sh
conda activate chicago

cd /home/aih/dinesh.haridoss/chicago_mil/patient_explorer

echo "Running on: $(hostname)"

# ── Reverse SSH tunnel (for on-campus / VPN access) ────────────────────────
ssh -o StrictHostKeyChecking=no \
    -o IdentitiesOnly=yes \
    -o ExitOnForwardFailure=no \
    -i /home/aih/dinesh.haridoss/.ssh/hpc_cluster.key \
    -fNR 8501:localhost:8501 hpc-submit01.scidom.de && \
    echo "SSH tunnel OK → hpc-submit01:8501" || \
    echo "SSH tunnel failed (Cloudflare will still work)"

# ── Cloudflare tunnel (public HTTPS URL for external collaborators) ─────────
CFDIR=/home/aih/dinesh.haridoss/.local/bin
mkdir -p "$CFDIR"
if [ ! -f "$CFDIR/cloudflared" ]; then
    echo "Downloading cloudflared..."
    wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
         -O "$CFDIR/cloudflared" && chmod +x "$CFDIR/cloudflared"
fi

# start cloudflare tunnel in background, capture URL from its log
CF_LOG=/tmp/cloudflared_$$.log
"$CFDIR/cloudflared" tunnel --url http://localhost:8501 \
    --no-autoupdate 2>"$CF_LOG" &
CF_PID=$!

# wait up to 20s for the public URL to appear
for i in $(seq 1 20); do
    CF_URL=$(grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
    [ -n "$CF_URL" ] && break
    sleep 1
done

echo ""
echo "════════════════════════════════════════"
if [ -n "$CF_URL" ]; then
    echo "  PUBLIC URL : $CF_URL"
    echo "  Password   : ${EXPLORER_PASSWORD:-lungmil2024}"
    echo "  Share both with collaborators."
else
    echo "  Cloudflare URL not found — check $CF_LOG"
fi
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

# cleanup
kill $CF_PID 2>/dev/null
