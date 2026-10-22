# Image for the pipeline's ECS Fargate tasks (infra/modules/pipeline_compute). Built and pushed to
# ECR by .github/workflows/pipeline-image.yml; the task runs e.g. `remote ingest --source esc50`.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

# The task definition sets VOCALOCATE_CONFIG (configs/<env>.toml), VOCALOCATE_LAKE_URI and
# VOCALOCATE_RAW_URI. /lake is the job's local working copy of the S3 lake.
RUN useradd --create-home pipeline && mkdir /lake && chown pipeline /lake
USER pipeline
ENTRYPOINT ["vocalocate-pipeline", "-v"]
CMD ["--help"]
