FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY guardrail ./guardrail
RUN pip install --no-cache-dir .
COPY policy.example.yaml ./policy.yaml
ENV GUARDRAIL_POLICY=/app/policy.yaml
EXPOSE 8787
# Set GUARDRAIL_API_KEYS at run time; point text_guard.url at your Ollama host.
CMD ["uvicorn", "guardrail.app:app", "--host", "0.0.0.0", "--port", "8787"]
