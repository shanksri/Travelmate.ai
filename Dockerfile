FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first, so code edits do not bust the layer cache.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend

RUN useradd --create-home --uid 1000 travelmate && chown -R travelmate /srv
USER travelmate

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Adds the test and lint tools. Used by the `tests` service in
# docker-compose.yml, which mounts the project over /srv.
FROM runtime AS dev
USER root
# From a directory holding only pyproject.toml, like the install above: with
# app/ and frontend/ beside it, setuptools refuses to guess which is the
# package ("Multiple top-level packages discovered in a flat-layout").
RUN mkdir /tmp/deps && cp /srv/pyproject.toml /tmp/deps/ \
    && pip install "/tmp/deps[dev]" && rm -rf /tmp/deps
USER travelmate
