FROM python:3.12-slim

LABEL maintainer="AAC Team"
LABEL description="Augmented Arbitrage Corp Trading Platform"

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create required directories
RUN mkdir -p logs data reports

# Non-root user
RUN useradd -m -r aac && chown -R aac:aac /app
USER aac

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080

ENTRYPOINT ["python", "launch.py"]
CMD ["health"]
