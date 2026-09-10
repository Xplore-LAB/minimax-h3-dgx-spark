<div align="center">

<img src="assets/icon.svg" alt="MiniMax H3 on DGX Spark" width="120" />

### 单台 DGX Spark / GB10 上容器化跑通 MiniMax H3 有声视频生成

为 DGX Spark / GB10（ARM64，128GB 统一内存）用户提供一套验证过的端到端部署：ComfyUI + MiniMax H3，文生视频 / 图生视频且带声音，宿主机保持干净。

[![快速开始](https://img.shields.io/badge/行动-快速开始-55e5d5?style=flat)](docs/2026-08-04_MiniMaxH3_第一条视频教程.md) [![故障排查](https://img.shields.io/badge/文档-故障排查-50a6ff?style=flat)](docs/故障排查.md) [![多段创作](https://img.shields.io/badge/教程-多段视频创作-9f86ff?style=flat)](docs/2026-08-04_MiniMaxH3_多段视频创作教程.md)

[![GitHub stars](https://img.shields.io/github/stars/Xplore-LAB/minimax-h3-dgx-spark?style=flat&label=stars&color=gold)](https://github.com/Xplore-LAB/minimax-h3-dgx-spark/stargazers)
[![license](https://img.shields.io/badge/license-MIT-1683c4?style=flat)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.30.0-32b643?style=flat)](https://github.com/comfyanonymous/ComfyUI)
[![Docker](https://img.shields.io/badge/Docker-compose-7e35d5?style=flat)](docker-compose.yml)

**简体中文** · [English](README.en.md)

</div>

---

## ⚡ 一分钟看懂 MiniMax H3 on DGX Spark

本仓库是在单台 NVIDIA DGX Spark / GB10（Blackwell SM121，aarch64，121-128GB 统一内存）上跑通 MiniMax H3 的完整方案：模型权重约 42.5 GB，全部负载跑在 Docker 容器（NGC PyTorch 25.11 + ComfyUI v0.30.0）里，模型、输入、输出挂在外部 NVMe。

| 你想完成的事 | 交付成果 |
|---|---|
| 部署 MiniMax H3 到 DGX Spark | 容器化环境 + 4 个必需模型文件（42.5 GB，断点续传下载） |
| 生成第一条带声音的视频 | HTTP API 全流程脚本，约 2-5 分钟出片 |
| 制作多段拼接长视频 | 8 段 × 5 秒拼成 40 秒大片的完整配方 |

## ✨ 核心能力

- **全容器化**：宿主机只装 DGX OS + 驱动 + Docker + NVIDIA Container Toolkit，ComfyUI 与依赖全部在容器内，保留 NGC 定制 torch 不覆盖
- **有声视频**：除视频 VAE（FP16，5.21 GB）外还带 Audio VAE（FP32，0.605 GB），音画联合输出
- **完整模型栈**：FL2VA pruned INT8 convrot 主模型（21.0 GB）+ Qwen3-VL-32B NVFP4 AWQ 文本编码器（15.7 GB）
- **工程化脚本**：`scripts/` 提供 init_storage、preflight、download、verify_models、status、collect_diagnostics 全流程工具
- **排障手册**：12 节故障排查文档 + 基准测试记录表 + 架构与优化说明

## 🚀 快速开始

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

### 4. 编译并固化 torchaudio（关键）

H3 音频 VAE 需要 `import torchaudio`，而 NGC torch（`2.10.0a0+b558c986e8.nv25.11`）与 PyPI 任何 torchaudio wheel 的 ABI 都不匹配，必须源码编译。NGC torch commit `b558c986e8` 缺 stable API 头文件，改用 2025-09-13 的 commit `87ff22e49ed0` 编译成功：

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

### 5. 生成第一条视频（HTTP API，带声音）

核心节点链：`UNETLoader(FL2VA) + CLIPLoader(Qwen3VL) + VAELoader×2(视频+音频) → MiniMaxH3SigmaShift → MiniMaxH3ImageToVideo → KSampler(20步, cfg=1.0) → VAEDecode + VAEDecodeAudio → CreateVideo/SaveVideo + SaveAudio`

完整可跑脚本见 [第一条视频教程](docs/2026-08-04_MiniMaxH3_第一条视频教程.md)，核心参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `length` | 96≈4s / 124≈5s / 362≈15s | 24fps，上限 15s |
| `width×height` | 608×352 ~ 1216×672 | 0.2MP ~ 0.8MP |
| `steps/cfg` | 20 / 1.0 | H3 官方推荐，euler/simple |
| 音画 | SaveVideo + SaveAudio | 输出后 loudnorm 归一化合成（电平默认 -34dB 听不见） |

图生视频：参考图放 `<NVMe>/minimax-h3/input/`，加 `LoadImage` 节点接到 `MiniMaxH3ImageToVideo.first_frame`。

## 📊 实测性能（GB10, 2026-08-04）

| 任务 | 分辨率 | 时长 | 耗时 |
|---|---:|---:|---:|
| 冒烟 | 608×352 | 4s | 96.8s |
| 常规 | 736×416 | 5s | 179.9s |
| 日常 | 864×480 | 5s | 255.8s（20 步） |
| 图生（竖） | 768×1152 | 5s | 15min 22s |
| 多段拼接 | 8×5s → 40s 大片 | — | ~50min（含合成拼接） |

规律：采样约 26-47s/步，耗时 ≈ 帧数 × 分辨率；竖屏/高分辨率显著变慢。

## 🛡️ 边界与注意

- 容器内任何安装必须 `docker commit` 固化进镜像，否则容器重建即丢
- `docker commit` 会连临时容器的入口命令一起固化，重建容器时检查 `docker inspect ... --format '{{.Config.Entrypoint}}'`，若不是 `/usr/bin/tini`，启动时显式加 `--entrypoint /usr/bin/tini ... -- /usr/local/bin/h3-entrypoint`
- 不要在容器里 pip install torch，CUDA 不可用时先 `make gpu-check` 排查驱动与容器工具
- 退出码 137 或系统卡顿说明统一内存压力打满，停掉其他大模型容器并降低分辨率
- 视频无声音说明 workflow 缺音频节点，需接 VAEDecodeAudio + SaveAudio
- 服务仅通过 SSH 隧道或可信局域网访问

## 📚 文档导航

- [第一条视频教程](docs/2026-08-04_MiniMaxH3_第一条视频教程.md)：HTTP API 全流程脚本
- [多段视频创作教程](docs/2026-08-04_MiniMaxH3_多段视频创作教程.md)：多段拼接长片配方
- [故障排查](docs/故障排查.md)：12 节完整排障手册
- [基准测试记录表](docs/基准测试记录表.md)：实测耗时记录
- [架构与优化说明](docs/架构与优化说明.md)：方案设计细节
- [提示词模板](docs/提示词模板.md)：视频生成提示词参考

## ✅ 完成标准

- [ ] `make gpu-check` 通过，8188 返回 200
- [ ] `docker exec minimax-h3-comfyui python -c "import torchaudio"` 通过（2.8.0a0）
- [ ] 608×352 / 736×416 / 864×480 三档连续生成成功
- [ ] 输出同时含视频与音频（音画联合）
- [ ] 图生视频（照片→动态）成功
- [ ] 服务仅通过 SSH 隧道或可信局域网访问

## 📄 许可证

- 本仓库代码：[MIT](LICENSE)
- MiniMax H3 模型：Community License（非 Apache 2.0），商用前请核查模型仓库 LICENSE 与所在地区限制
