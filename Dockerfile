FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY specmodel ./specmodel
RUN pip install --no-cache-dir .
COPY export ./export

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python -m specmodel serve --export export/* --host 0.0.0.0 --port ${PORT}"]
