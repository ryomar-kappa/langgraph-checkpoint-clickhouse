FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY pyproject.toml constraints-test.txt README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install \
    --index-url https://pypi.org/simple \
    --constraint constraints-test.txt \
    --no-cache-dir \
    ".[test]"

CMD ["python", "-m", "pytest", "-q"]
