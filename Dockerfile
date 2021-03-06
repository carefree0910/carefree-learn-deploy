FROM python:3.8-slim AS builder
WORKDIR /carefree-learn-deploy
COPY . core
RUN python -m venv .venv &&  \
    .venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    .venv/bin/pip install -U pip setuptools numpy cython && \
    cd core && ../.venv/bin/pip install . && cd .. && \
    rm -rf core && \
    find /carefree-learn-deploy/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM nvcr.io/nvidia/tritonserver:22.04-py3
WORKDIR /carefree-learn-deploy
COPY --from=builder /carefree-learn-deploy /carefree-learn-deploy
ENV PATH="/carefree-learn-deploy/.venv/bin:$PATH"
ENV PYTHONPATH="/carefree-learn-deploy/.venv/lib/python3.8/site-packages:$PYTHONPATH"

EXPOSE 8000
CMD ["tritonserver", "--model-repository", "/carefree-learn-deploy/apis/models"]