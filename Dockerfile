ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ARG COMFYUI_REF=v0.30.0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_XET_HIGH_PERFORMANCE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates ffmpeg \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      tini procps jq \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# 源码在宿主机预下载(codeload), 避免容器内 GitHub 直连 TLS 断连
COPY comfyui-src/ /opt/ComfyUI/

WORKDIR /opt/ComfyUI

# 保留 NGC 镜像内经过验证的 CUDA/PyTorch；禁止 requirements 覆盖 torch。
RUN grep -Ev '^(torch|torchvision|torchaudio)([<=>~! ].*)?$' requirements.txt \
      > /tmp/comfy-requirements.txt \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/comfy-requirements.txt \
    && python -m pip install "huggingface_hub[hf_xet]>=0.34.0"

COPY docker/entrypoint.sh /usr/local/bin/h3-entrypoint
COPY docker/download_models.py /usr/local/bin/h3-download-models
RUN chmod +x /usr/local/bin/h3-entrypoint /usr/local/bin/h3-download-models

EXPOSE 8188
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/h3-entrypoint"]
