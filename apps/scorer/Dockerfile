FROM python:3.11-slim

WORKDIR /app

COPY apps/scorer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/scorer/worker.py .
COPY apps/scorer/model_runtime.py .
COPY apps/scorer/online_state.py .
COPY ml ./ml

ENV PYTHONPATH=/app

CMD ["python","-u", "worker.py"]
