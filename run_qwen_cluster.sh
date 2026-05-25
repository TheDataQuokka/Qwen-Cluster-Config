#!/usr/bin/env bash
# =============================================================================
# run_qwen_cluster.sh
# Qwen3.6-35B-A3B MTP Load-Balanced Cluster — A100 80GB
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗ FATAL:${NC} $*"; exit 1; }

# ── Configurable defaults (override via env or qwen-cluster.conf) ─────────────
LLAMA_DIR="${LLAMA_DIR:-/opt/llama}"
MODEL_DIR="${MODEL_DIR:-/opt/models/qwen3.6-35b-mtp}"
CONF_FILE="${CONF_FILE:-/opt/qwen-cluster.conf}"
PORTS="${PORTS:-8081 8082}"
LB_PORT="${LB_PORT:-8000}"
CONTEXT="${CONTEXT:-32768}"
DRAFT_N="${DRAFT_N:-3}"
KV_TYPE="${KV_TYPE:-q8_0}"
API_KEY="${API_KEY:-}"
THINKING_DEFAULT="${THINKING_DEFAULT:-false}"
MODEL_REPO="unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MMPROJ_FILE="mmproj-F16.gguf"
HF_TOKEN="${HF_TOKEN:-}"

# ── Load existing conf if present ────────────────────────────────────────────
[[ -f "$CONF_FILE" ]] && { log "Loading config from $CONF_FILE"; source "$CONF_FILE"; }

# =============================================================================
# PHASE 1 — PREFLIGHT
# =============================================================================
phase1_preflight() {
  echo -e "\n${BOLD}━━━ PHASE 1: PREFLIGHT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  [[ "$EUID" -ne 0 ]] && die "Run as root or with sudo."

  # CUDA / GPU check
  command -v nvidia-smi &>/dev/null || die "nvidia-smi not found — is CUDA installed?"
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  ok "GPU detected: ${GPU_NAME} (${GPU_VRAM} MiB)"

  VRAM_GB=$(( GPU_VRAM / 1024 ))
  if (( VRAM_GB < 40 )); then
    warn "GPU has only ${VRAM_GB} GB VRAM. This script targets 80 GB A100."
    warn "Continuing anyway — you may need to reduce to 1 instance or lower context."
    read -rp "Continue? [y/N] " yn; [[ "$yn" =~ ^[Yy]$ ]] || exit 0
  fi

  # Packages
  log "Installing system dependencies..."
  apt-get update -qq
  apt-get install -y -qq git curl nginx python3-pip pciutils build-essential > /dev/null
  ok "System packages ready"

  # huggingface-cli
  if ! command -v huggingface-cli &>/dev/null; then
    log "Installing huggingface-hub..."
    pip install -q huggingface-hub hf_transfer
    ok "huggingface-cli installed"
  else
    ok "huggingface-cli already present"
  fi

  # HF_HUB_ENABLE_HF_TRANSFER for fast downloads
  export HF_HUB_ENABLE_HF_TRANSFER=1
}

# =============================================================================
# PHASE 2 — BINARY (prebuilt, no compile)
# =============================================================================
phase2_binary() {
  echo -e "\n${BOLD}━━━ PHASE 2: LLAMA.CPP BINARY ━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  if [[ -x "$LLAMA_DIR/llama-server" ]]; then
    CURRENT_VER=$("$LLAMA_DIR/llama-server" --version 2>&1 | grep -oP 'version: \K[0-9]+' || echo "unknown")
    ok "llama-server already installed (build ${CURRENT_VER})"

    # Check if MTP support is present (build >= ~9100 for merged MTP)
    if [[ "$CURRENT_VER" != "unknown" ]] && (( CURRENT_VER >= 9100 )); then
      ok "MTP support confirmed (build ${CURRENT_VER} >= 9100)"
      return 0
    else
      warn "Installed build ${CURRENT_VER} may predate MTP merge (need >= 9100). Re-downloading."
    fi
  fi

  # Query ai-dock/llama.cpp-cuda for latest release
  log "Querying ai-dock/llama.cpp-cuda for latest CUDA 12 amd64 binary..."
  RELEASE_JSON=$(curl -fsSL "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest")
  DOWNLOAD_URL=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assets = data.get('assets', [])
# Prefer cuda-12.8 amd64, fall back to cuda-12.6
for tag in ['cuda-12.8-amd64', 'cuda-12.6-amd64', 'cuda-12']:
    for a in assets:
        if tag in a['name'] and 'amd64' in a['name']:
            print(a['browser_download_url'])
            sys.exit(0)
# last resort: first amd64 tarball
for a in assets:
    if 'amd64' in a['name'] and a['name'].endswith('.tar.gz'):
        print(a['browser_download_url'])
        sys.exit(0)
print('')
")

  if [[ -z "$DOWNLOAD_URL" ]]; then
    die "Could not find a suitable llama.cpp CUDA binary from ai-dock/llama.cpp-cuda.\nCheck https://github.com/ai-dock/llama.cpp-cuda/releases manually."
  fi

  TARBALL=$(basename "$DOWNLOAD_URL")
  log "Downloading: $TARBALL"
  mkdir -p "$LLAMA_DIR"
  TMP_DIR=$(mktemp -d)
  curl -fL --progress-bar "$DOWNLOAD_URL" -o "$TMP_DIR/$TARBALL"

  log "Extracting..."
  tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

  # Find and copy binaries — directory name varies by release
  BIN_SRC=$(find "$TMP_DIR" -name "llama-server" -type f | head -1)
  [[ -z "$BIN_SRC" ]] && die "llama-server not found in tarball."
  BIN_DIR=$(dirname "$BIN_SRC")

  cp "$BIN_DIR"/llama-* "$LLAMA_DIR/" 2>/dev/null || true
  # Copy shared libs if present
  LIB_SRC=$(dirname "$BIN_DIR")/lib
  [[ -d "$LIB_SRC" ]] && cp -r "$LIB_SRC"/* "$LLAMA_DIR/" 2>/dev/null || true
  # Some builds bundle libs next to binaries
  cp "$BIN_DIR"/*.so* "$LLAMA_DIR/" 2>/dev/null || true

  chmod +x "$LLAMA_DIR"/llama-*
  rm -rf "$TMP_DIR"

  # Verify CUDA backend loads
  LVER=$("$LLAMA_DIR/llama-server" --version 2>&1 || true)
  if echo "$LVER" | grep -q "ggml_cuda_init"; then
    ok "llama-server CUDA verified: $(echo "$LVER" | grep 'version:' || echo "$LVER" | head -1)"
  else
    warn "CUDA init message not seen in --version output. Attempting test load..."
    # Quick sanity — if binary runs at all we're probably fine
    if "$LLAMA_DIR/llama-server" --help &>/dev/null; then
      warn "Binary runs but CUDA init unclear. Proceeding — check logs on first start."
    else
      die "llama-server binary failed to run. Check $LLAMA_DIR."
    fi
  fi
}

# =============================================================================
# PHASE 3 — MODEL DOWNLOAD
# =============================================================================
phase3_model() {
  echo -e "\n${BOLD}━━━ PHASE 3: MODEL DOWNLOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  FULL_MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
  FULL_MMPROJ_PATH="$MODEL_DIR/$MMPROJ_FILE"
  EXPECTED_SIZE_GB=22

  if [[ -f "$FULL_MODEL_PATH" ]]; then
    ACTUAL_GB=$(( $(stat -c%s "$FULL_MODEL_PATH") / 1024 / 1024 / 1024 ))
    if (( ACTUAL_GB >= EXPECTED_SIZE_GB )); then
      ok "Model already present (${ACTUAL_GB} GB) — skipping download"
      [[ -f "$FULL_MMPROJ_PATH" ]] && ok "mmproj already present" || warn "mmproj missing — re-downloading"
      [[ -f "$FULL_MMPROJ_PATH" ]] && return 0
    else
      warn "Model file found but appears incomplete (${ACTUAL_GB} GB < ${EXPECTED_SIZE_GB} GB). Re-downloading."
    fi
  fi

  mkdir -p "$MODEL_DIR"
  log "Downloading $MODEL_REPO — this will take a while (~23 GB)..."
  log "Using hf_transfer for maximum speed."

  HF_ARGS=(
    huggingface-cli download "$MODEL_REPO"
    --local-dir "$MODEL_DIR"
    --include "*${MODEL_FILE##*-}"   # match *UD-Q4_K_XL.gguf
    --include "*Q4_K_XL*"
    --include "*mmproj*"
    --local-dir-use-symlinks False
  )
  # Simpler: just include exact filenames
  HF_ARGS=(
    huggingface-cli download "$MODEL_REPO"
    --local-dir "$MODEL_DIR"
    --include "$MODEL_FILE"
    --include "$MMPROJ_FILE"
    --local-dir-use-symlinks False
  )

  [[ -n "$HF_TOKEN" ]] && HF_ARGS+=(--token "$HF_TOKEN")

  "${HF_ARGS[@]}" || die "Model download failed. Check your network or set HF_TOKEN if needed."

  [[ -f "$FULL_MODEL_PATH" ]] || die "Model file not found after download: $FULL_MODEL_PATH"
  [[ -f "$FULL_MMPROJ_PATH" ]] || warn "mmproj not found at $FULL_MMPROJ_PATH — will run without vision support."

  FINAL_GB=$(( $(stat -c%s "$FULL_MODEL_PATH") / 1024 / 1024 / 1024 ))
  ok "Model ready: $FULL_MODEL_PATH (${FINAL_GB} GB)"
}

# =============================================================================
# PHASE 4 — WRITE CONFIG FILE
# =============================================================================
phase4_conf() {
  echo -e "\n${BOLD}━━━ PHASE 4: CONFIG FILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  cat > "$CONF_FILE" << EOF
# Qwen3.6 Cluster Config — auto-generated $(date)
# Edit and re-run the script to apply changes.

LLAMA_DIR="$LLAMA_DIR"
MODEL_DIR="$MODEL_DIR"
MODEL_FILE="$MODEL_FILE"
MMPROJ_FILE="$MMPROJ_FILE"
PORTS="$PORTS"
LB_PORT="$LB_PORT"
CONTEXT="$CONTEXT"
DRAFT_N="$DRAFT_N"
KV_TYPE="$KV_TYPE"
API_KEY="$API_KEY"
THINKING_DEFAULT="$THINKING_DEFAULT"
HF_TOKEN="$HF_TOKEN"
EOF

  chmod 600 "$CONF_FILE"
  ok "Config written to $CONF_FILE"
}

# =============================================================================
# PHASE 5 — SYSTEMD UNITS
# =============================================================================
phase5_systemd() {
  echo -e "\n${BOLD}━━━ PHASE 5: SYSTEMD UNITS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  MMPROJ_LINE=""
  MMPROJ_PATH="$MODEL_DIR/$MMPROJ_FILE"
  [[ -f "$MMPROJ_PATH" ]] && MMPROJ_LINE="  --mmproj ${MMPROJ_PATH} \\"

  API_KEY_LINE=""
  [[ -n "$API_KEY" ]] && API_KEY_LINE="  --api-key ${API_KEY} \\"

  THINKING_KWARGS='{"enable_thinking":false}'
  [[ "$THINKING_DEFAULT" == "true" ]] && THINKING_KWARGS='{"enable_thinking":true}'

  cat > /etc/systemd/system/qwen@.service << EOF
[Unit]
Description=Qwen3.6-35B-A3B MTP llama-server on port %i
After=network.target
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
EnvironmentFile=$CONF_FILE
Environment=LD_LIBRARY_PATH=${LLAMA_DIR}
ExecStart=${LLAMA_DIR}/llama-server \\
  -m ${MODEL_DIR}/${MODEL_FILE} \\
${MMPROJ_LINE}
  --spec-type draft-mtp \\
  --spec-draft-n-max ${DRAFT_N} \\
  -c ${CONTEXT} \\
  --parallel 1 \\
  -ngl 999 \\
  --cache-type-k ${KV_TYPE} \\
  --cache-type-v ${KV_TYPE} \\
  --flash-attn \\
  --host 127.0.0.1 --port %i \\
  --temp 0.6 \\
  --top-p 0.95 \\
  --top-k 20 \\
  --min-p 0.0 \\
  --presence-penalty 1.5 \\
  --jinja \\
  --chat-template-kwargs '${THINKING_KWARGS}' \\
${API_KEY_LINE}
  --log-format text
Restart=on-failure
RestartSec=15
TimeoutStartSec=300
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload

  # Stop any running instances first
  for port in $PORTS; do
    systemctl stop "qwen@${port}.service" 2>/dev/null || true
  done

  log "Starting llama-server instances..."
  for port in $PORTS; do
    systemctl enable --now "qwen@${port}.service"
    log "Started qwen@${port}"
  done

  # Wait for health endpoints
  log "Waiting for instances to become healthy (model load ~60s)..."
  for port in $PORTS; do
    ATTEMPTS=0
    while (( ATTEMPTS < 60 )); do
      if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        ok "qwen@${port} is healthy"
        break
      fi
      sleep 3
      (( ATTEMPTS++ ))
      (( ATTEMPTS % 10 == 0 )) && log "  still waiting for port ${port} (${ATTEMPTS}× checks)..."
    done
    if (( ATTEMPTS >= 60 )); then
      warn "qwen@${port} did not become healthy in time. Check: journalctl -u qwen@${port} -n 50"
    fi
  done
}

# =============================================================================
# PHASE 6 — NGINX
# =============================================================================
phase6_nginx() {
  echo -e "\n${BOLD}━━━ PHASE 6: NGINX LOAD BALANCER ━━━━━━━━━━━━━━━━━━━━━━${NC}"

  # Build upstream block from PORTS
  UPSTREAM_SERVERS=""
  for port in $PORTS; do
    UPSTREAM_SERVERS+="        server 127.0.0.1:${port} max_fails=2 fail_timeout=10s;\n"
  done

  cat > /etc/nginx/sites-available/qwen-cluster << EOF
upstream qwen_backend {
    least_conn;
${UPSTREAM_SERVERS}
    # Reuse connections — avoids TCP handshake per request
    keepalive 8;
    keepalive_requests 1000;
    keepalive_timeout 60s;
}

server {
    listen ${LB_PORT};
    server_name _;

    # Large client body for long prompts
    client_max_body_size 64m;
    client_body_timeout 60s;

    location / {
        proxy_pass http://qwen_backend;

        # Required for keepalive to upstream
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;

        # Streaming — MUST be off for SSE/token streaming
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_cache off;

        # Timeouts — 600s covers thinking mode long outputs
        proxy_connect_timeout 10s;
        proxy_send_timeout    600s;
        proxy_read_timeout    600s;
    }

    location /health {
        proxy_pass http://qwen_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        access_log off;
    }
}
EOF

  # Enable site
  ln -sf /etc/nginx/sites-available/qwen-cluster /etc/nginx/sites-enabled/qwen-cluster
  # Disable default site to avoid port conflict
  rm -f /etc/nginx/sites-enabled/default

  nginx -t || die "nginx config test failed."
  systemctl enable --now nginx
  systemctl reload nginx
  ok "nginx configured and reloaded"
}

# =============================================================================
# PHASE 7 — STATUS REPORT
# =============================================================================
phase7_status() {
  echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  Qwen3.6-35B-A3B MTP Cluster — Ready${NC}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

  # VRAM usage
  echo -e "${CYAN}GPU VRAM:${NC}"
  nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total \
    --format=csv,noheader | awk -F', ' '{
      printf "  %-40s used: %s  free: %s  total: %s\n", $1, $2, $3, $4
    }'
  echo ""

  # Instance status
  echo -e "${CYAN}Instances:${NC}"
  for port in $PORTS; do
    STATUS=$(systemctl is-active "qwen@${port}" 2>/dev/null || echo "unknown")
    HEALTH=$(curl -sf "http://127.0.0.1:${port}/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "unreachable")
    if [[ "$STATUS" == "active" ]]; then
      echo -e "  ${GREEN}●${NC} qwen@${port}  systemd: ${STATUS}  health: ${HEALTH}"
    else
      echo -e "  ${RED}●${NC} qwen@${port}  systemd: ${STATUS}  health: ${HEALTH}"
    fi
  done
  echo ""

  # Connection info
  HOST_IP=$(hostname -I | awk '{print $1}')
  echo -e "${CYAN}Endpoints:${NC}"
  echo -e "  Load balancer:   http://${HOST_IP}:${LB_PORT}"
  echo -e "  Chat completions: http://${HOST_IP}:${LB_PORT}/v1/chat/completions"
  echo -e "  Models:           http://${HOST_IP}:${LB_PORT}/v1/models"
  echo ""

  echo -e "${CYAN}Quick test:${NC}"
  if [[ -n "$API_KEY" ]]; then
    echo -e "  curl http://${HOST_IP}:${LB_PORT}/v1/chat/completions \\"
    echo -e "    -H 'Authorization: Bearer ${API_KEY}' \\"
    echo -e "    -H 'Content-Type: application/json' \\"
    echo -e "    -d '{\"model\":\"qwen3.6\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}'"
  else
    echo -e "  curl http://${HOST_IP}:${LB_PORT}/v1/chat/completions \\"
    echo -e "    -H 'Content-Type: application/json' \\"
    echo -e "    -d '{\"model\":\"qwen3.6\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}'"
  fi
  echo ""

  echo -e "${CYAN}Useful commands:${NC}"
  echo -e "  journalctl -u qwen@8081 -f          # tail instance logs"
  echo -e "  journalctl -u qwen@8082 -f"
  echo -e "  systemctl restart qwen@8081          # restart one instance"
  echo -e "  systemctl restart 'qwen@*'           # restart all"
  echo -e "  nginx -t && systemctl reload nginx   # reload nginx"
  echo -e "  cat $CONF_FILE                  # view config"
  echo ""
  echo -e "${CYAN}Config:${NC}"
  echo -e "  Context: ${CONTEXT} tokens  |  KV: ${KV_TYPE}  |  MTP draft: ${DRAFT_N}  |  Thinking: ${THINKING_DEFAULT}"
  [[ -n "$API_KEY" ]] && echo -e "  API key: ${API_KEY}"
  echo ""
}

# =============================================================================
# MAIN
# =============================================================================
main() {
  echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗"
  echo -e "║   Qwen3.6-35B-A3B MTP Cluster Setup                 ║"
  echo -e "║   Target: A100 80GB Ubuntu                          ║"
  echo -e "╚══════════════════════════════════════════════════════╝${NC}\n"

  phase1_preflight
  phase2_binary
  phase3_model
  phase4_conf
  phase5_systemd
  phase6_nginx
  phase7_status
}

main "$@"
