<div align="center">

<img src="assets/icon.svg" alt="MiniMax H3 on DGX Spark" width="120" />

### Run MiniMax H3 video generation with sound on a single DGX Spark / GB10, fully containerized

A verified end-to-end deployment of ComfyUI + MiniMax H3 (text-to-video / image-to-video, with audio) on NVIDIA DGX Spark / GB10 (ARM64, 128GB unified memory), while keeping the host clean.

[![Quick Start](https://img.shields.io/badge/action-quick%20start-55e5d5?style=flat)](docs/2026-08-04_MiniMaxH3_第一条视频教程.md) [![Troubleshooting](https://img.shields.io/badge/docs-troubleshooting-50a6ff?style=flat)](docs/故障排查.md) [![Multi-segment Guide](https://img.shields.io/badge/guide-multi--segment%20film-9f86ff?style=flat)](docs/2026-08-04_MiniMaxH3_多段视频创作教程.md)

[![GitHub stars](https://img.shields.io/github/stars/Xplore-LAB/minimax-h3-dgx-spark?style=flat&label=stars&color=gold)](https://github.com/Xplore-LAB/minimax-h3-dgx-spark/stargazers)
[![license](https://img.shields.io/badge/license-MIT-1683c4?style=flat)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.30.0-32b643?style=flat)](https://github.com/comfyanonymous/ComfyUI)
[![Docker](https://img.shields.io/badge/Docker-compose-7e35d5?style=flat)](docker-compose.yml)

[简体中文](README.md) · **English**

</div>

---

## ⚡ MiniMax H3 on DGX Spark in One Minute

This repo is a complete recipe for running MiniMax H3 on a single NVIDIA DGX Spark / GB10 (Blackwell SM121, aarch64, 121-128GB unified memory). Weights total about 42.5 GB, everything runs inside Docker (NGC PyTorch 25.11 + ComfyUI v0.30.0), and models, inputs, and outputs live on an external NVMe drive.

| What you want to do | What you get |
|---|---|
| Deploy MiniMax H3 on DGX Spark | Containerized environment + 4 required model files (42.5 GB, resumable download) |
| Generate your first video with sound | Full HTTP API workflow, about 2-5 minutes per clip |
| Produce a multi-segment film | Complete recipe for stitching 8 x 5s segments into a 40s film |

## ✨ Key Capabilities

- **Fully containerized**: the host only needs DGX OS, drivers, Docker, and NVIDIA Container Toolkit; ComfyUI and all dependencies live in the container, and the NGC-customized torch is kept untouched
- **Video with sound**: besides the video VAE (FP16, 5.21 GB) there is an audio VAE (FP32, 0.605 GB) for joint audio-video output
- **Full model stack**: FL2VA pruned INT8 convrot main model (21.0 GB) + Qwen3-VL-32B NVFP4 AWQ text encoder (15.7 GB)
- **Engineering scripts**: `scripts/` ships init_storage, preflight, download, verify_models, status, and collect_diagnostics tooling
- **Troubleshooting handbook**: a 12-section troubleshooting doc plus benchmark log and architecture notes

## 🚀 Quick Start

### 1. Fetch the ComfyUI source (this repo does not vendor upstream code)

```bash
./scripts/prepare_comfyui_src.sh    # downloads v0.30.0 via codeload into comfyui-src/
```

### 2. Initialize + check

```bash
make init DATA_ROOT=/mnt/<your-NVMe>/minimax-h3   # generates .env and creates directories
make preflight
make gpu-check     # should print torch.cuda.is_available() = True, GPU: NVIDIA GB10
```

⚠️ Do not run other large-model containers (qwen/vllm, etc.) in parallel while generating video; unified memory will be exhausted.

### 3. Build + download models + launch

```bash
make build
make download       # only the 4 required files (42.5 GB, resumable)
make verify-models
make up && make logs    # wait for "To see the GUI go to: http://0.0.0.0:8188"
curl http://127.0.0.1:8188/system_stats   # returns 200
```

If HuggingFace connections drop, add `HF_ENDPOINT=https://hf-mirror.com` to `.env`; set `HF_TOKEN` when authentication is required (never commit it to git).

### 4. Compile and bake in torchaudio (critical)

The H3 audio VAE requires `import torchaudio`, but the NGC torch (`2.10.0a0+b558c986e8.nv25.11`) matches no PyPI torchaudio wheel ABI, so a source build is mandatory. The NGC torch commit `b558c986e8` lacks stable API headers; the 2025-09-13 commit `87ff22e49ed0` builds successfully:

```bash
curl -L -o /tmp/torchaudio-src.tar.gz \
  https://codeload.github.com/pytorch/audio/tar.gz/87ff22e49ed0
docker cp /tmp/torchaudio-src.tar.gz minimax-h3-comfyui:/tmp/
docker exec minimax-h3-comfyui bash -c '
  cd /tmp && tar xzf torchaudio-src.tar.gz
  pip install --no-build-isolation --no-deps /tmp/audio-87ff22e49ed0
  python -c "import torchaudio; print(torchaudio.__version__)"   # → 2.8.0a0
'
docker commit minimax-h3-comfyui local/minimax-h3-comfyui:v0.30.0-ta   # bake it in!
```

### 5. Generate your first video (HTTP API, with sound)

Core node chain: `UNETLoader(FL2VA) + CLIPLoader(Qwen3VL) + VAELoader x2 (video + audio) → MiniMaxH3SigmaShift → MiniMaxH3ImageToVideo → KSampler (20 steps, cfg=1.0) → VAEDecode + VAEDecodeAudio → CreateVideo/SaveVideo + SaveAudio`

A full runnable script is in the [first-video tutorial](docs/2026-08-04_MiniMaxH3_第一条视频教程.md). Key parameters:

| Parameter | Value | Notes |
|---|---|---|
| `length` | 96≈4s / 124≈5s / 362≈15s | 24fps, 15s cap |
| `width×height` | 608×352 ~ 1216×672 | 0.2MP ~ 0.8MP |
| `steps/cfg` | 20 / 1.0 | H3 official recommendation, euler/simple |
| A/V | SaveVideo + SaveAudio | loudnorm-normalized after export (default level -34dB is inaudible) |

Image-to-video: place the reference image in `<NVMe>/minimax-h3/input/` and wire a `LoadImage` node into `MiniMaxH3ImageToVideo.first_frame`.

## 📊 Measured Performance (GB10, 2026-08-04)

| Task | Resolution | Duration | Time |
|---|---:|---:|---:|
| Smoke test | 608×352 | 4s | 96.8s |
| Standard | 736×416 | 5s | 179.9s |
| Daily use | 864×480 | 5s | 255.8s (20 steps) |
| I2V (portrait) | 768×1152 | 5s | 15min 22s |
| Multi-segment | 8×5s → 40s film | — | ~50min (incl. compositing) |

Pattern: sampling runs about 26-47s per step; time scales with frames × resolution; portrait/high resolutions are significantly slower.

## 🛡️ Boundaries and Caveats

- Any in-container installation must be baked into the image with `docker commit`, or it is lost when the container is recreated
- `docker commit` also bakes in the temporary container entrypoint; when recreating, check `docker inspect ... --format '{{.Config.Entrypoint}}'`, and if it is not `/usr/bin/tini`, start with an explicit `--entrypoint /usr/bin/tini ... -- /usr/local/bin/h3-entrypoint`
- Never pip install torch inside the container; if CUDA is unavailable, run `make gpu-check` to triage drivers and the container toolkit first
- Exit code 137 or system stalls mean unified memory pressure; stop other large-model containers and lower the resolution
- Silent video means the workflow lacks audio nodes; wire in VAEDecodeAudio + SaveAudio
- Expose the service only through an SSH tunnel or a trusted LAN

## 📚 Documentation

- [First-video tutorial](docs/2026-08-04_MiniMaxH3_第一条视频教程.md): full HTTP API workflow script
- [Multi-segment film tutorial](docs/2026-08-04_MiniMaxH3_多段视频创作教程.md): stitching recipe for longer films
- [Troubleshooting](docs/故障排查.md): the full 12-section handbook
- [Benchmark log](docs/基准测试记录表.md): measured runtimes
- [Architecture and optimization notes](docs/架构与优化说明.md): design details
- [Prompt templates](docs/提示词模板.md): prompt references for video generation

Note: the docs are currently written in Chinese.

## ✅ Definition of Done

- [ ] `make gpu-check` passes, port 8188 returns 200
- [ ] `docker exec minimax-h3-comfyui python -c "import torchaudio"` works (2.8.0a0)
- [ ] Three consecutive generations succeed at 608×352 / 736×416 / 864×480
- [ ] Output contains both video and audio (joint A/V)
- [ ] Image-to-video (photo to motion) works
- [ ] Service is reachable only via SSH tunnel or trusted LAN

## 📄 License

- Code in this repo: [MIT](LICENSE)
- MiniMax H3 model: Community License (not Apache 2.0); check the model repo LICENSE and your local regulations before commercial use
