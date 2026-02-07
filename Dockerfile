# ============================================================
# Accelerated Arbitrage Corp - Docker Configuration
# ============================================================
# Multi-stage build for optimized production image

FROM python:3.12-slim as builder

# Build arguments
ARG ENVIRONMENT=production

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
COPY TradingExecution/requirements.txt ./TradingExecution/
COPY CentralAccounting/requirements.txt ./CentralAccounting/
COPY BigBrainIntelligence/requirements.txt ./BigBrainIntelligence/
COPY shared/requirements.txt ./shared/
COPY tests/requirements.txt ./tests/

# Install Python dependencies
RUN pip install --no-cache-dir --user \
    -r requirements.txt \
    -r TradingExecution/requirements.txt \
    -r CentralAccounting/requirements.txt \
    -r BigBrainIntelligence/requirements.txt \
    -r shared/requirements.txt

# ============================================================
# Production Image
# ============================================================
FROM python:3.12-slim as production

# Labels
LABEL maintainer="Accelerated Arbitrage Corp"
LABEL version="1.0.0"
LABEL description="Multi-theater arbitrage trading system"

# Create non-root user for security
RUN groupadd -r acc && useradd -r -g acc acc

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/acc/.local

# Copy application code
COPY --chown=acc:acc . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R acc:acc /app/data /app/logs

# Set environment variables
ENV PATH=/home/acc/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER acc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from shared.config_loader import get_config; c = get_config(); print('OK')" || exit 1

# Default command
CMD ["python", "main.py", "--paper"]

# ============================================================
# Development Image (alternative target)
# ============================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy

USER acc

# Development command with auto-reload capability
CMD ["python", "main.py", "--paper", "--dry-run"]
