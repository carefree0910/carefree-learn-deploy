FROM python:3.8-slim AS builder
WORKDIR /carefree-learn-deploy
COPY . .
RUN python -m venv .venv &&  \
    .venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    .venv/bin/pip install -U pip setuptools numpy cython && \
    .venv/bin/pip install . && \
    find /carefree-learn-deploy/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM nvcr.io/nvidia/tritonserver:21.09-py3
WORKDIR /carefree-learn-deploy
COPY --from=builder /carefree-learn-deploy /carefree-learn-deploy
ENV PATH="/carefree-learn-deploy/.venv/bin:$PATH"
ENV PYTHONPATH="/carefree-learn-deploy/.venv/lib/python3.8/site-packages:$PYTHONPATH"
COPY apis apis

EXPOSE 8000
CMD ["tritonserver", "--model-repository", "/carefree-learn-deploy/apis/models"]