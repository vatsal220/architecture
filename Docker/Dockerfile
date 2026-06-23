# =============================================================================
# Toronto Data Platform - Recommendation Pipeline (training / batch image)
# -----------------------------------------------------------------------------
# A single image that runs the full training + batch-scoring pipeline. It is the
# unit of promotion in the target platform: pushed to ECR and executed as a
# Kubernetes Job (orchestrated by Argo Workflows) and a CronJob (nightly batch)
# on Amazon EKS. "Runs on my laptop" and "runs in Prod" are the same artifact;
# only the entry-point arguments + the pod's IAM role (IRSA) differ.
# The real-time API uses a sibling image, Docker/Dockerfile.serving.
# Build context is the repository root:  docker build -f Docker/Dockerfile .
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source, configs and entry point.
COPY src ./src
COPY configs ./configs
COPY run_pipeline.py ./

# Non-root user (least privilege; mirrors the platform's container hardening).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--config", "configs/pipeline.yaml"]
