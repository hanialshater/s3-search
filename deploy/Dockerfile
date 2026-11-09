# One image for the replicas, the indexer Job and the operator. Build from the repository root:
#
#   docker build -f deploy/Dockerfile -t s3-search:cpu .                          # CPU (numpy reference), the default
#   docker build -f deploy/Dockerfile --build-arg BASE=cuda -t s3-search:cuda .   # CUDA: torch with CUDA, requirements-gpu.txt
#
# BASE picks the base image and the requirements file; the code and the entry points are the same, and the device is
# configuration (S3_SEARCH_DEVICE, set by the operator from spec.device). Building is local; nothing is pushed.
ARG BASE=cpu

FROM python:3.12-slim AS base-cpu
ENV S3_REQUIREMENTS=requirements.txt

# torch and its CUDA runtime come with the base image, so requirements-gpu.txt finds torch already installed.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base-cuda
ENV S3_REQUIREMENTS=requirements-gpu.txt

FROM base-${BASE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
# pytest is a test dependency (requirements.txt lists it); it is small, and keeps `python -m pytest` usable in the image.
# Optional BuildKit secret `cacert`: a CA bundle for a TLS-inspecting proxy (--secret id=cacert,src=/path/ca.pem).
# Absent in CI, where pip uses the system roots; never stored in an image layer.
RUN --mount=type=secret,id=cacert,required=false \
    if [ -f /run/secrets/cacert ]; then export PIP_CERT=/run/secrets/cacert; fi; \
    pip install -r "${S3_REQUIREMENTS}"
COPY engine ./engine
COPY dataplane ./dataplane
COPY serving ./serving
COPY deploy ./deploy
# /data is the segment volume (a PersistentVolumeClaim in the cluster); the pods run as 10001 with fsGroup 10001.
RUN mkdir -p /data && chown 10001:10001 /data
USER 10001
ENV S3_SERVING_DATA_DIR=/data S3_SEARCH_DEVICE=cpu
EXPOSE 8080
CMD ["python", "-m", "serving"]
