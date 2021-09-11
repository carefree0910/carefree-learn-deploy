FROM python:3.6-slim

# conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# conda
RUN conda create -n cflearn_deploy python=3.6
SHELL ["conda", "run", "-n", "cflearn_deploy", "/bin/bash", "-c"]

# faiss
RUN conda install -c pytorch faiss-cpu

# main
WORKDIR /carefree-learn-deploy
COPY . .
RUN pip install .
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "80"]