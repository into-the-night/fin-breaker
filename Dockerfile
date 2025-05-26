# Dockerfile for fin-breaker
# Build and run all services
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["streamlit", "run", "streamlit_app/app.py"]
