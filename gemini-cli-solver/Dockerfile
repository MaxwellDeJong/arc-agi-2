FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy scipy matplotlib pandas \
    scikit-learn scikit-image pillow \
    sympy networkx \
    openai

RUN mkdir -p /workspace /app && chmod 777 /workspace /app
