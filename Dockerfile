## ── Build stage ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System build dependencies (gcc for C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual env
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

## ── Runtime stage ────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="AAC Team"
LABEL description="Augmented Arbitrage Corp Trading Platform"

WORKDIR /app

# Copy only the pre-built virtual env (no gcc in runtime image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application code
COPY . .

# Create required directories
RUN mkdir -p logs data reports

# Non-root user
RUN useradd -m -r aac && chown -R aac:aac /app
USER aac

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080

ENTRYPOINT ["python", "launch.py"]
CMD ["health"]
