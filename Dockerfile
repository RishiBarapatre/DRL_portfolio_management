# syntax=docker/dockerfile:1
# (README Addition: Parser directive to enable modern BuildKit features)

# ==========================================
# STAGE 1: The Builder (The "Workshop")
# ==========================================
# STAGE 1: The Builder
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# (README Addition: Create a virtual environment to isolate dependencies)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into the venv
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: The Runtime (The "Living Room")
# ==========================================
# STAGE 2: The Runtime
FROM python:3.10-slim

# Set environment variables and put the venv on the system PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# (README Addition: Create a non-root user for security)
RUN groupadd -g 1000 appuser && useradd -m -u 1000 -g appuser appuser

WORKDIR /usr/src/app

# (README Addition: Use COPY --link to speed up layer caching)
# Copy the built virtual environment from Stage 1
COPY --link --from=builder /opt/venv /opt/venv

# Copy the application code and explicitly grant ownership to our non-root user
COPY --link --chown=1000:1000 . .

# Ensure the persistent data directories exist and are owned by the non-root user
RUN mkdir -p project_data training_outputs && \
    chown -R 1000:1000 project_data training_outputs

# (README Addition: Switch away from root)
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]