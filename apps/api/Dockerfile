FROM python:3.11-slim
WORKDIR /app

COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/app.py /app/app.py
COPY apps/scripts /app/scripts

EXPOSE 8000
CMD ["python", "app.py"]
