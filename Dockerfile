FROM python:3.10-slim
WORKDIR /
COPY requirements.txt .
COPY Fastapi.py .
COPY bestmodel.joblib .
RUN pip install -r requirements.txt



ENTRYPOINT ["uvicorn", "Fastapi:app", "--host", "0.0.0.0", "--port", "8004"]