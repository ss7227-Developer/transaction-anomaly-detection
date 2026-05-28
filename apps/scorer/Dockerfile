FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/apps/scorer/
COPY ../../ml /app/ml/
WORKDIR /app/apps/scorer
ENV PYTHONPATH=/app
CMD ["python", "worker.py"]
