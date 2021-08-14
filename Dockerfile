FROM python:3.6-slim
WORKDIR /usr/home
COPY . .
RUN pip install . --default-timeout=10000
RUN pip install -r requirements.txt --default-timeout=10000
EXPOSE 80
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "80"]