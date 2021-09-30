FROM python:3.6-slim AS builder
WORKDIR /carefree-learn-deploy
COPY . .
RUN python -m venv .venv &&  \
    .venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    .venv/bin/pip install -U pip setuptools numpy cython && \
    .venv/bin/pip install . && \
    .venv/bin/pip install -r requirements.txt && \
    find /carefree-learn-deploy/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM python:3.6-slim
WORKDIR /carefree-learn-deploy
COPY --from=builder /carefree-learn-deploy /carefree-learn-deploy
ENV PATH="/carefree-learn-deploy/.venv/bin:$PATH"
COPY apis apis

EXPOSE 80
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "80"]