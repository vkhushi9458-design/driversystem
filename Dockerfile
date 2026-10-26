
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/data/driver_monitor.db

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /data \
    && chown app:app /data

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app driver_monitor.py web_app.py ./
COPY --chown=app:app templates/ ./templates/

USER app

EXPOSE 5000
VOLUME ["/data"]

CMD ["python", "web_app.py"]
