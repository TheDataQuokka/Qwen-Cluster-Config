# qwen-cluster

One-script setup for a **Qwen3.6-35B-A3B MTP** load-balanced inference cluster on a single A100 80GB. Curl it onto any fresh Ubuntu GPU instance and have a production-ready OpenAI-compatible endpoint running in under 10 minutes (model download aside).

```bash
curl -fsSL https://raw.githubusercontent.com/TheDataQuokka/Qwen-Cluster-Config/main/run_qwen_cluster.sh | sudo bash
```

---

## What it does

| Phase | Action |
|---|---|
| 1. Preflight | Checks CUDA/GPU, installs `nginx`, `huggingface-cli`, system deps |
| 2. Binary | Pulls latest prebuilt llama.cpp CUDA binary — no compile step |
| 3. Model | Downloads `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-Q4_K_XL via `hf_transfer` |
| 4. Config | Writes `/opt/qwen-cluster.conf` (persists settings across runs) |
| 5. Systemd | Installs `qwen@.service` template unit, starts two instances, polls `/health` |
| 6. nginx | Configures `least_conn` load balancer with streaming-safe settings |
| 7. Status | Prints VRAM usage, instance health, curl example, useful commands |

On completion you have:

- Two `llama-server` instances on `127.0.0.1:8081` and `:8082`, managed by systemd (survive SSH disconnect, restart on reboot)
- nginx on `:8000` load-balancing across them
- OpenAI-compatible API at `http://<host>:8000/v1`

---

## Requirements

- Ubuntu 22.04 / 24.04
- NVIDIA A100 80GB (see [other GPUs](#other-gpus))
- CUDA drivers already installed (`nvidia-smi` must work)
- ~25 GB free disk space for the model
- Internet access (HuggingFace + GitHub)

---

## VRAM budget

```
UD-Q4_K_XL  ×2 instances  =  45.8 GB  model weights
KV cache (q8_0, 32K ctx)  =  ~17 GB   per instance  →  34 GB total
                              ──────────────────────────────────────
Total                      ≈  80 GB   (fits an A100 80GB)
```

The model is MoE — only ~3B parameters activate per token — so two full copies coexist on the GPU without the active compute overhead you'd expect from a dense 35B model.

> **Why 2 instances and not 3?**  
> The MTP GGUF is 22.9 GB (not 17.7 GB). Three copies = 68.7 GB, leaving only ~11 GB for KV cache across all three. Not enough for reasonable context at q8_0. Two instances is the safe number for 80 GB.

---

## MTP (Multi-Token Prediction)

This setup uses Qwen3.6's built-in MTP heads for speculative decoding — the model drafts 3 tokens ahead and verifies them in one pass, giving roughly **1.5–2× faster generation** with no quality loss.

Key constraint: **MTP requires `--parallel 1`** per instance. You cannot run multiple concurrent slots with MTP enabled — each instance handles one request at a time. Concurrency comes from having two instances behind nginx.

---

## Configuration

Settings live in `/opt/qwen-cluster.conf`. Edit and re-run the script to apply changes.

```bash
LLAMA_DIR="/opt/llama"
MODEL_DIR="/opt/models/qwen3.6-35b-mtp"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MMPROJ_FILE="mmproj-F16.gguf"
PORTS="8081 8082"
LB_PORT="8000"
CONTEXT="32768"           # tokens per instance; reduce to 16384 to free KV cache
DRAFT_N="3"               # MTP draft tokens; 3 is the sweet spot
KV_TYPE="q8_0"            # q4_0 frees ~8 GB if you need more headroom
API_KEY=""                # set to require bearer auth
THINKING_DEFAULT="false"  # true = thinking mode on by default
HF_TOKEN=""               # only needed for gated models
```

### Override at runtime

```bash
# With API key
API_KEY=mysecretkey sudo bash run_qwen_cluster.sh

# With HuggingFace token (if needed)
HF_TOKEN=hf_xxx sudo bash run_qwen_cluster.sh

# Custom context
CONTEXT=16384 sudo bash run_qwen_cluster.sh
```

---

## Usage

The endpoint is OpenAI-compatible. Point any client at `http://<host>:8000`.

**Streaming completion:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "explain VWAP in 3 sentences"}],
    "stream": true
  }'
```

**With thinking mode** (prepend `/think` to trigger, or set `THINKING_DEFAULT=true`):
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "/think solve this step by step: ..."}],
    "stream": true
  }'
```

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://<host>:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3.6",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Useful commands

```bash
# Instance logs
journalctl -u qwen@8081 -f
journalctl -u qwen@8082 -f

# Restart one instance
systemctl restart qwen@8081

# Restart all instances
systemctl restart 'qwen@*'

# Check health
curl -s http://localhost:8081/health
curl -s http://localhost:8082/health

# VRAM usage
nvidia-smi

# nginx status / reload
systemctl status nginx
nginx -t && systemctl reload nginx

# View config
cat /opt/qwen-cluster.conf
```

---

## Re-running the script

The script is idempotent:

- **Binary** — skips if a recent build (≥9100) is already present
- **Model** — skips if file is present and full size
- **Config** — overwrites with current settings
- **Systemd** — stops, reconfigures, and restarts instances
- **nginx** — reconfigures and reloads

Re-run any time to update settings from `qwen-cluster.conf` or after editing env vars.

---

## Other GPUs

The script warns but continues on GPUs with < 40 GB VRAM. Adjust accordingly:

| GPU | Recommendation |
|---|---|
| A100 40GB | 1 instance, reduce context to 16384 |
| RTX 4090 24GB | 1 instance, UD-Q3_K_XL quant (~17 GB), context 8192 |
| 2× A100 40GB | Set `PORTS="8081 8082"`, use `CUDA_VISIBLE_DEVICES` per unit |
| H100 80GB | Same as A100 80GB — should work identically |

For single-instance setups just set `PORTS="8081"` and nginx will proxy to that one instance.

---

## Model details

| | |
|---|---|
| Model | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| GGUF source | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) |
| Quant | UD-Q4_K_XL (Unsloth Dynamic 2.0, importance-matrix calibrated) |
| File size | 22.9 GB |
| Architecture | MoE, 35B total / ~3B active params per token |
| MTP heads | Yes (required for `--spec-type draft-mtp`) |
| Context (native) | 262K |
| Context (this setup) | 32K default (safe for q8_0 KV on 80 GB) |

---

## Repo structure

```
qwen-cluster/
├── run_qwen_cluster.sh   # the script
└── README.md             # this file
```

---

## License

MIT****
