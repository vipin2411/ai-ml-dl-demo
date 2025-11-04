# ai_app/Dockerfile
FROM python:3.9-slim-buster
WORKDIR /app
COPY app.py .
RUN pip install Flask
CMD ["python", "app.py"]
