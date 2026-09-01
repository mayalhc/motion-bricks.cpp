FROM python:3.10.14-slim-bookworm@sha256:2407c61b1a18067393fecd8a22cf6fceede893b6aaca817bf9fbfe65e33614a3

RUN python -m pip install --no-cache-dir --upgrade pip==24.3.1 \
    && python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.0 \
    && python -m pip install --no-cache-dir \
      numpy==1.26.4 \
      PyYAML==6.0.2 \
      safetensors==0.4.5

WORKDIR /work
ENTRYPOINT ["python", "/work/reference/extract_safe.py"]
