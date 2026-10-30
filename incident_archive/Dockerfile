FROM vllm/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635

USER root
ENV PYTHONNOUSERSITE=1 UV_CACHE_DIR=/tmp/uv-cache
RUN uv pip uninstall --system mooncake-transfer-engine mooncake-transfer-engine-cuda13 && \
    uv pip install --system --no-deps 'https://files.pythonhosted.org/packages/b0/2b/5c24b1b33b08bb2ee4e2408b2354f67598590ba6ec2607b92117061643e6/mooncake_transfer_engine_cuda13-0.3.13.post1-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=155bcb1b02d07f4b78268641880ada0558abeac1e5db0624f879f35bba4c8cd6'
COPY vllm/ /opt/incident-source/vllm/
COPY incident_archive/ /opt/incident-source/incident_archive/
RUN uv run --no-project --no-sync python /opt/incident-source/incident_archive/install_overlay.py \
    --source-root /opt/incident-source --apply-offline
ENV VLLM_ISOLATED_OBSERVABILITY=0 VLLM_COMPAT_OBSERVABILITY=0
LABEL archive.vllm="0.28.0" archive.mooncake="0.3.13.post1-cuda13" \
      archive.incident="2026-09-18" archive.post-incident-lifetime-fixes="absent"
