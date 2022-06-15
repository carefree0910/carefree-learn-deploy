FROM nvcr.io/nvidia/tritonserver:22.04-py3
WORKDIR /carefree-learn-deploy

EXPOSE 8000
CMD ["tritonserver", "--model-repository", "/models"]