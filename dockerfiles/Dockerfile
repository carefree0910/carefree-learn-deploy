FROM python:3.6-slim
WORKDIR /carefree-learn-deploy
COPY . .
RUN pip install .
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "80"]