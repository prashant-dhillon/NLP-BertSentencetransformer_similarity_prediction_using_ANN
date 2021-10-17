FROM python:3.7-slim-stretch

ENV PORT 5000

COPY ./src /src

COPY requirements.txt /src/requirements.txt

WORKDIR /src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT
RUN echo "#!/bin/bash \n gunicorn app:app -w 3 -b :${PORT} -t 360 --log-level ERROR --timeout 10000" > ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]