FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn requests lxml
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY app.py /app/app.py
COPY xmla_result_parser.py /app/xmla_result_parser.py
COPY session_ui.html /app/session_ui.html

ENV UPSTREAM_XMLA=http://emondrian:8080/emondrian/xmla
ENV MCAD_EVAL_URL=http://mcad-api:8000/eval
ENV MCAD_CKG_URL=http://mcad-api:8000/ckg/update
ENV MCAD_API_BASE=http://mcad-api:8000

EXPOSE 9000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
