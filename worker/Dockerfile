# SUSE BCI per lab container policy
FROM registry.suse.com/bci/python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest.py .
USER 1000
ENTRYPOINT ["python3", "ingest.py"]
