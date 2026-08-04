<div align="center">

# 🎬 MiniMax H3 on DGX Spark / GB10

**Run MiniMax H3 (text/video → video with sound) on a single NVIDIA DGX Spark / GB10, fully containerized**
在单台 DGX Spark / GB10（128GB 统一内存）上跑通 MiniMax H3 文生视频 / 图生视频（**带声音**），宿主机保持干净、全部容器化。

[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.30.0-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

**English**: A verified end-to-end deployment of [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (FL2VA INT8 + Qwen3-VL-32B NVFP4 text encoder + video/audio VAE, ~42.5 GB weights) with ComfyUI v0.30.0 on DGX Spark/GB10 (ARM64, unified memory). Everything runs inside Docker on NGC PyTorch 25.11 — the host stays clean. Includes the full HTTP API workflow to generate your first video with sound in ~2–5 minutes, plus multi-segment film production recipes.

</div>

---

## 方案总览

```text
DGX Spark / GB10
├── 宿主机：DGX OS + NVIDIA 驱动 + Docker + NVIDIA Container Toolkit
├── 容器：NGC PyTorch 25.11（torch 2.10.0a0+nv25.11）+ ComfyUI v0.30.0
├── H3 主模型：FL2VA pruned INT8 convrot（21.0 GB）
├── 文本编码器：Qwen3-VL-32B NVFP4 AWQ（15.7 GB）
├── Video VAE：FP16（5.21 GB）
└── Audio VAE：FP32（0.605 GB）—— 带声音的关键
```

**核心决策**
- 宿主机保持干净，ComfyUI 与依赖全部容器化；模型/输入/输出挂在外部 NVMe
- 保留 NGC 定制 torch，**不覆盖**；torchaudio 扩展必须源码编译匹配 ABI（见 §4.4）
- 容器内任何安装必须 `docker commit` 固化进镜像，否则容器重建即丢

## 环境要求

| 项目 | 要求 |
|---|---|
| 服务器 | DGX Spark / GB10（Blackwell SM121，aarch64） |
| 内存 | 121–128 GB 统一内存 |
| 存储 | 外部 NVMe ≥ 100GB（模型 42.5 GB + 输出空间） |
| 软件 | Docker + NVIDIA Container Toolkit，DGX OS 6.x/7.x |
| 网络 | HuggingFace 走 hf-mirror.com 镜像；GitHub 源码走 codeload（国内直连易断） |

## 目录结构

```text
<NVMe>/minimax-h3/                ← H3_DATA_ROOT
├── models/{diffusion_models, text_encoders, vae}/
├── input/                        ← 图生视频参考图
├── output/                       ← 成片（+ audio/ 子目录）
├── temp/  user/  custom_nodes/  hf-cache/
```

仓库（本目录）：
```text
Dockerfile  docker-compose.yml  Makefile  .env.example
docker/{entrypoint.sh, download_models.py}
scripts/{init_storage, preflight, prepare_comfyui_src, download_fl2va, verify_models, status, collect_diagnostics}.sh
docs/{第一条视频教程, 多段视频创作教程, 故障排查, 基准测试记录表, 架构与优化说明, 提示词模板}.md
```

## 快速开始

### 1. 准备 ComfyUI 源码（仓库不携带上游代码）

```bash
./scripts/prepare_comfyui_src.sh    # codeload 下载 v0.30.0 到 comfyui-src/
```

### 2. 初始化 + 检查

```bash
make init DATA_ROOT=/mnt/<你的NVMe>/minimax-h3   # 生成 .env 并建目录
make preflight
make gpu-check     # 应输出 torch.cuda.is_available() = True, GPU: NVIDIA GB10
```

⚠️ 生成视频时不要并行跑其他大模型容器（qwen/vllm 等），统一内存会被打满。

### 3. 构建 + 下载模型 + 启动

```bash
make build
make download       # 只下 4 个必需文件（42.5 GB，断点续传）
make verify-models
make up && make logs    # 看到 "To see the GUI go to: http://0.0.0.0:8188"
curl http://127.0.0.1:8188/system_stats   # 返回 200
```

HuggingFace 断连：`.env` 加 `HF_ENDPOINT=https://hf-mirror.com`；需认证时填 `HF_TOKEN`（勿提交 git）。

### 4. 编译并固化 torchaudio（关键！）

> 为什么必须做：H3 音频 VAE 需要 `import torchaudio`，而 NGC torch（`2.10.0a0+b558c986e8.nv25.11`）
> 与 PyPI 任何 torchaudio wheel ABI 都不匹配（报 `undefined symbol: torch_get_mutable_data_ptr`），必须源码编译。
> NGC torch commit `b558c986e8` 缺 stable API 头文件，**改用 2025-09-13 的 commit `87ff22e49ed0` 编译成功**。

```bash
curl -L -o /tmp/torchaudio-src.tar.gz \
  https://codeload.github.com/pytorch/audio/tar.gz/87ff22e49ed0
docker cp /tmp/torchaudio-src.tar.gz minimax-h3-comfyui:/tmp/
docker exec minimax-h3-comfyui bash -c '
  cd /tmp && tar xzf torchaudio-src.tar.gz
  pip install --no-build-isolation --no-deps /tmp/audio-87ff22e49ed0
  python -c "import torchaudio; print(torchaudio.__version__)"   # → 2.8.0a0
'
docker commit minimax-h3-comfyui local/minimax-h3-comfyui:v0.30.0-ta   # 固化！
```

⚠️ **commit 陷阱**：commit 会连临时容器的入口命令一起固化。重建容器时检查
`docker inspect ... --format '{{.Config.Entrypoint}}'`，若不是 `/usr/bin/tini`，
启动时显式加 `--entrypoint /usr/bin/tini ... -- /usr/local/bin/h3-entrypoint`。

## 生成你的第一条视频（HTTP API，带声音）

核心节点链：`UNETLoader(FL2VA) + CLIPLoader(Qwen3VL) + VAELoader×2(视频+音频) → MiniMaxH3SigmaShift → MiniMaxH3ImageToVideo → KSampler(20步, cfg=1.0) → VAEDecode + VAEDecodeAudio → CreateVideo/SaveVideo + SaveAudio`

完整可跑脚本见 [`docs/2026-08-04_MiniMaxH3_第一条视频教程.md`](docs/2026-08-04_MiniMaxH3_第一条视频教程.md)，核心参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `length` | 96≈4s / 124≈5s / 362≈15s | 24fps，上限 15s |
| `width×height` | 608×352 ~ 1216×672 | 0.2MP ~ 0.8MP |
| `steps/cfg` | 20 / 1.0 | H3 官方推荐，euler/simple |
| 音画 | SaveVideo + SaveAudio | 输出后 loudnorm 归一化合成（电平默认 -34dB 听不见） |

图生视频：参考图放 `<NVMe>/minimax-h3/input/`，加 `LoadImage` 节点接到 `MiniMaxH3ImageToVideo.first_frame`。

## 实测性能（GB10, 2026-08-04）

| 任务 | 分辨率 | 时长 | 耗时 |
|---|---:|---:|---:|
| 冒烟 | 608×352 | 4s | 96.8s |
| 常规 | 736×416 | 5s | 179.9s |
| 日常 | 864×480 | 5s | 255.8s（20 步） |
| 图生（竖） | 768×1152 | 5s | 15min 22s |
| 多段拼接 | 8×5s → 40s 大片 | — | ~50min（含合成拼接） |

规律：采样约 26–47s/步，耗时 ≈ 帧数 × 分辨率；竖屏/高分辨率显著变慢。

## 故障排查速查

| 症状 | 原因 / 处理 |
|---|---|
| 容器 Exited，报 `No module named 'torchaudio'` | 可写层丢失 → 重装并 `docker commit`（§4.4） |
| `undefined symbol: torch_get_mutable_data_ptr` | PyPI wheel ABI 不匹配 → 源码编译 87ff22e49ed0 |
| 新容器只 sleep 不启动 | commit 固化了入口 → `--entrypoint /usr/bin/tini` 覆盖 |
| `torch.cuda.is_available()=False` | 驱动/容器工具 → `make gpu-check`，勿在容器 pip install torch |
| 退出码 137 / 系统卡顿 | 统一内存压力 → 停其他大模型容器，降分辨率 |
| 视频无声音 | workflow 缺音频节点 → 接 VAEDecodeAudio + SaveAudio |
| GitHub/HF 下载断连 | 用 hf-mirror.com / codeload 镜像 |

完整 12 节排查见 [`docs/故障排查.md`](docs/故障排查.md)。

## 完成标准

- [ ] `make gpu-check` 通过，8188 返回 200
- [ ] `docker exec minimax-h3-comfyui python -c "import torchaudio"` 通过（2.8.0a0）
- [ ] 608×352 / 736×416 / 864×480 三档连续生成成功
- [ ] 输出同时含视频与音频（音画联合）
- [ ] 图生视频（照片→动态）成功
- [ ] 服务仅通过 SSH 隧道或可信局域网访问

## License

- 本仓库代码：MIT
- **MiniMax H3 模型：Community License（非 Apache 2.0）**——商用前请核查模型仓库 LICENSE 与所在地区限制
