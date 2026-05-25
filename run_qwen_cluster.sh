#!/usr/bin/env bash
# =============================================================================
# run_qwen_cluster.sh
# Qwen3.6-35B-A3B MTP Load-Balanced Cluster — A100 80GB
# Docker-based — uses ghcr.io/ggml-org/llama.cpp:server-cuda13
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
die()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗ FATAL:${NC} $*"; exit 1; }

# ── Configurable defaults (override via env or conf file) ─────────────────────
MODEL_DIR="${MODEL_DIR:-/opt/models/qwen3.6-35b-mtp}"
CONF_FILE="${CONF_FILE:-/opt/qwen-cluster.conf}"
PORTS="${PORTS:-8081 8082}"
LB_PORT="${LB_PORT:-8000}"
CONTEXT="${CONTEXT:-32768}"
DRAFT_N="${DRAFT_N:-3}"
KV_TYPE="${KV_TYPE:-q8_0}"
API_KEY="${API_KEY:-}"
THINKING_DEFAULT="${THINKING_DEFAULT:-false}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
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

  # GPU check
  command -v nvidia-smi &>/dev/null || die "nvidia-smi not found — is the NVIDIA driver installed?"
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  ok "GPU detected: ${GPU_NAME} (${GPU_VRAM} MiB)"

  VRAM_GB=$(( GPU_VRAM / 1024 ))
  if (( VRAM_GB < 40 )); then
    warn "GPU has only ${VRAM_GB} GB VRAM. This script targets 80 GB A100."
    warn "Continuing anyway — you may need to reduce to 1 instance or lower context."
    read -rp "Continue? [y/N] " yn; [[ "$yn" =~ ^[Yy]$ ]] || exit 0
  fi

  # System packages
  log "Installing system dependencies..."
  apt-get update -qq
  apt-get install -y -qq curl nginx python3-pip unzip > /dev/null
  ok "System packages ready"

  # Docker
  if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh > /dev/null 2>&1
    ok "Docker installed"
  else
    ok "Docker already present ($(docker --version | cut -d' ' -f3 | tr -d ','))"
  fi

  # Pull the llama.cpp image (idempotent — skips if up to date)
  log "Pulling Docker image: ${DOCKER_IMAGE}..."
  docker pull "$DOCKER_IMAGE" | tail -1
  ok "Docker image ready"

  # Verify GPU passthrough works in Docker
  log "Verifying GPU passthrough..."
  DOCKER_VER=$(docker run --rm --device nvidia.com/gpu=all "$DOCKER_IMAGE" --version 2>&1 | grep 'version:' || true)
  if [[ -n "$DOCKER_VER" ]]; then
    ok "GPU passthrough verified: $DOCKER_VER"
  else
    die "Docker GPU passthrough failed. Check: docker run --rm --device nvidia.com/gpu=all $DOCKER_IMAGE --version"
  fi

  # hf CLI (huggingface_hub)
  if ! command -v hf &>/dev/null; then
    log "Installing huggingface-hub..."
    pip install -q --root-user-action=ignore huggingface-hub
    ok "hf CLI installed"
  else
    ok "hf CLI already present"
  fi

  # Enable Xet high-performance transfer
  export HF_XET_HIGH_PERFORMANCE=1
}

# =============================================================================
# PHASE 2 — MODEL DOWNLOAD
# =============================================================================
phase2_model() {
  echo -e "\n${BOLD}━━━ PHASE 2: MODEL DOWNLOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  FULL_MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
  FULL_MMPROJ_PATH="$MODEL_DIR/$MMPROJ_FILE"
  EXPECTED_SIZE_GB=22

  if [[ -f "$FULL_MODEL_PATH" ]]; then
    ACTUAL_GB=$(( $(stat -c%s "$FULL_MODEL_PATH") / 1024 / 1024 / 1024 ))
    if (( ACTUAL_GB >= EXPECTED_SIZE_GB )); then
      ok "Model already present (${ACTUAL_GB} GB) — skipping download"
      [[ -f "$FULL_MMPROJ_PATH" ]] && ok "mmproj already present" && return 0
      warn "mmproj missing — re-downloading"
    else
      warn "Model file found but appears incomplete (${ACTUAL_GB} GB < ${EXPECTED_SIZE_GB} GB). Re-downloading."
    fi
  fi

  mkdir -p "$MODEL_DIR"
  log "Downloading $MODEL_REPO (~23 GB) using hf_transfer..."

  HF_ARGS=(
    hf download "$MODEL_REPO"
    --local-dir "$MODEL_DIR"
    --include "$MODEL_FILE"
    --include "$MMPROJ_FILE"
    --local-dir-use-symlinks False
  )
  [[ -n "$HF_TOKEN" ]] && HF_ARGS+=(--token "$HF_TOKEN")

  HF_XET_HIGH_PERFORMANCE=1 "${HF_ARGS[@]}" || die "Model download failed. Check network or set HF_TOKEN if needed."

  [[ -f "$FULL_MODEL_PATH" ]] || die "Model file not found after download: $FULL_MODEL_PATH"
  [[ -f "$FULL_MMPROJ_PATH" ]] || warn "mmproj not found — will run without vision support."

  FINAL_GB=$(( $(stat -c%s "$FULL_MODEL_PATH") / 1024 / 1024 / 1024 ))
  ok "Model ready: $FULL_MODEL_PATH (${FINAL_GB} GB)"
}

# =============================================================================
# PHASE 3 — WRITE CONFIG FILE
# =============================================================================
phase3_conf() {
  echo -e "\n${BOLD}━━━ PHASE 3: CONFIG FILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  cat > "$CONF_FILE" << EOF
# Qwen3.6 Cluster Config — generated $(date)
# Edit and re-run the script to apply changes.

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
DOCKER_IMAGE="$DOCKER_IMAGE"
HF_TOKEN="$HF_TOKEN"
EOF

  chmod 600 "$CONF_FILE"
  ok "Config written to $CONF_FILE"
}

# =============================================================================
# PHASE 4 — SYSTEMD UNITS (docker run)
# =============================================================================
phase4_systemd() {
  echo -e "\n${BOLD}━━━ PHASE 4: SYSTEMD UNITS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  MMPROJ_ARG=""
  [[ -f "$MODEL_DIR/$MMPROJ_FILE" ]] && MMPROJ_ARG="--mmproj /models/${MMPROJ_FILE}"

  API_KEY_ARG=""
  [[ -n "$API_KEY" ]] && API_KEY_ARG="--api-key ${API_KEY}"

  THINKING_KWARGS='{"enable_thinking":false}'
  [[ "$THINKING_DEFAULT" == "true" ]] && THINKING_KWARGS='{"enable_thinking":true}'

  cat > /etc/systemd/system/qwen@.service << EOF
[Unit]
Description=Qwen3.6-35B-A3B MTP llama-server (Docker) on port %i
After=network.target docker.service
Requires=docker.service
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
EnvironmentFile=${CONF_FILE}
# Remove existing container if it exists (e.g. after crash)
ExecStartPre=-/usr/bin/docker rm -f qwen-%i
ExecStart=/usr/bin/docker run --rm \\
  --name qwen-%i \\
  --device nvidia.com/gpu=all \\
  -v \${MODEL_DIR}:/models:ro \\
  -p 127.0.0.1:%i:8080 \\
  \${DOCKER_IMAGE} \\
  -m /models/\${MODEL_FILE} \\
  ${MMPROJ_ARG} \\
  --spec-type draft-mtp \\
  --spec-draft-n-max \${DRAFT_N} \\
  -c \${CONTEXT} \\
  --parallel 1 \\
  -ngl 999 \\
  --cache-type-k \${KV_TYPE} \\
  --cache-type-v \${KV_TYPE} \\
  --flash-attn \\
  --host 0.0.0.0 --port 8080 \\
  --temp 0.6 \\
  --top-p 0.95 \\
  --top-k 20 \\
  --min-p 0.0 \\
  --presence-penalty 1.5 \\
  --jinja \\
  --chat-template-kwargs '${THINKING_KWARGS}' \\
  ${API_KEY_ARG} \\
  --log-format text
ExecStop=/usr/bin/docker stop qwen-%i
Restart=on-failure
RestartSec=15
TimeoutStartSec=300
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload

  # Stop any running instances first
  for port in $PORTS; do
    systemctl stop "qwen@${port}.service" 2>/dev/null || true
    docker rm -f "qwen-${port}" 2>/dev/null || true
  done

  log "Starting instances..."
  for port in $PORTS; do
    systemctl enable --now "qwen@${port}.service"
    log "Started qwen@${port}"
  done

  # Wait for health endpoints
  log "Waiting for instances to become healthy (model load ~60-90s)..."
  for port in $PORTS; do
    ATTEMPTS=0
    while (( ATTEMPTS < 90 )); do
      if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        ok "qwen@${port} is healthy"
        break
      fi
      sleep 3
      (( ATTEMPTS++ ))
      (( ATTEMPTS % 10 == 0 )) && log "  still waiting for port ${port} (${ATTEMPTS}× checks)..."
    done
    if (( ATTEMPTS >= 90 )); then
      warn "qwen@${port} did not become healthy in time."
      warn "Check logs: journalctl -u qwen@${port} -n 50"
      warn "Or: docker logs qwen-${port}"
    fi
  done
}

# =============================================================================
# PHASE 5 — NGINX
# =============================================================================
phase5_nginx() {
  echo -e "\n${BOLD}━━━ PHASE 5: NGINX LOAD BALANCER ━━━━━━━━━━━━━━━━━━━━━━${NC}"

  UPSTREAM_SERVERS=""
  for port in $PORTS; do
    UPSTREAM_SERVERS+="        server 127.0.0.1:${port} max_fails=2 fail_timeout=10s;\n"
  done

  cat > /etc/nginx/sites-available/qwen-cluster << EOF
upstream qwen_backend {
    least_conn;
${UPSTREAM_SERVERS}
    keepalive 8;
    keepalive_requests 1000;
    keepalive_timeout 60s;
}

server {
    listen ${LB_PORT};
    server_name _;

    client_max_body_size 64m;
    client_body_timeout 60s;

    location / {
        proxy_pass http://qwen_backend;

        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;

        # Critical for SSE/token streaming
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_cache off;

        # 600s covers thinking mode long outputs
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

  ln -sf /etc/nginx/sites-available/qwen-cluster /etc/nginx/sites-enabled/qwen-cluster
  rm -f /etc/nginx/sites-enabled/default

  nginx -t || die "nginx config test failed."
  systemctl enable --now nginx
  systemctl reload nginx
  ok "nginx configured and reloaded"
}

# =============================================================================
# PHASE 6 — STATUS REPORT
# =============================================================================
phase6_status() {
  echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  Qwen3.6-35B-A3B MTP Cluster — Ready${NC}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

  echo -e "${CYAN}GPU VRAM:${NC}"
  nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total \
    --format=csv,noheader | awk -F', ' '{
      printf "  %-40s used: %s  free: %s  total: %s\n", $1, $2, $3, $4
    }'
  echo ""

  echo -e "${CYAN}Instances:${NC}"
  for port in $PORTS; do
    STATUS=$(systemctl is-active "qwen@${port}" 2>/dev/null || echo "unknown")
    HEALTH=$(curl -sf "http://127.0.0.1:${port}/health" 2>/dev/null | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "unreachable")
    CONTAINER=$(docker ps --filter "name=qwen-${port}" --format "{{.Status}}" 2>/dev/null || echo "unknown")
    if [[ "$STATUS" == "active" ]]; then
      echo -e "  ${GREEN}●${NC} qwen@${port}  systemd: ${STATUS}  health: ${HEALTH}  container: ${CONTAINER}"
    else
      echo -e "  ${RED}●${NC} qwen@${port}  systemd: ${STATUS}  health: ${HEALTH}  container: ${CONTAINER}"
    fi
  done
  echo ""

  HOST_IP=$(hostname -I | awk '{print $1}')
  echo -e "${CYAN}Endpoints:${NC}"
  echo -e "  Load balancer:    http://${HOST_IP}:${LB_PORT}"
  echo -e "  Chat completions: http://${HOST_IP}:${LB_PORT}/v1/chat/completions"
  echo -e "  Models:           http://${HOST_IP}:${LB_PORT}/v1/models"
  echo ""

  echo -e "${CYAN}Quick test:${NC}"
  AUTH=""
  [[ -n "$API_KEY" ]] && AUTH="-H 'Authorization: Bearer ${API_KEY}' \\"
  echo -e "  curl http://localhost:${LB_PORT}/v1/chat/completions \\"
  [[ -n "$API_KEY" ]] && echo -e "    -H 'Authorization: Bearer ${API_KEY}' \\"
  echo -e "    -H 'Content-Type: application/json' \\"
  echo -e "    -d '{\"model\":\"qwen3.6\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}'"
  echo ""

  echo -e "${CYAN}Useful commands:${NC}"
  echo -e "  journalctl -u qwen@8081 -f        # systemd logs instance 1"
  echo -e "  journalctl -u qwen@8082 -f        # systemd logs instance 2"
  echo -e "  docker logs -f qwen-8081          # container logs instance 1"
  echo -e "  docker logs -f qwen-8082          # container logs instance 2"
  echo -e "  systemctl restart qwen@8081       # restart one instance"
  echo -e "  systemctl restart 'qwen@*'        # restart all"
  echo -e "  docker pull ${DOCKER_IMAGE}  # update image"
  echo -e "  nginx -t && systemctl reload nginx"
  echo ""

  echo -e "${CYAN}Config:${NC}"
  echo -e "  Context: ${CONTEXT} tokens  |  KV: ${KV_TYPE}  |  MTP draft: ${DRAFT_N}  |  Thinking: ${THINKING_DEFAULT}"
  echo -e "  Image: ${DOCKER_IMAGE}"
  [[ -n "$API_KEY" ]] && echo -e "  API key: ${API_KEY}"
  echo -e "  Config file: ${CONF_FILE}"
  echo ""
}

# =============================================================================
# MAIN
# =============================================================================
main() {
  echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗"
  echo -e "║   Qwen3.6-35B-A3B MTP Cluster Setup                 ║"
  echo -e "║   Docker + nginx — A100 80GB Ubuntu                 ║"
  echo -e "╚══════════════════════════════════════════════════════╝${NC}\n"

  phase1_preflight
  phase2_model
  phase3_conf
  phase4_systemd
  phase5_nginx
  phase6_status
}

main "$@"
