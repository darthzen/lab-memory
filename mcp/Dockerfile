# SUSE BCI per lab container policy
FROM registry.suse.com/bci/python:3.12

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

ENV PYTHONPATH=/app/src \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8765

USER 1000
EXPOSE 8765
ENTRYPOINT ["python3", "-m", "lab_memory_mcp.server"]
