FROM python:3.8.3

ENV CONFIG_DIR=/var/code

WORKDIR ${CONFIG_DIR}

USER root

COPY . ${CONFIG_DIR}

RUN pip install --upgrade pip && \
    pip install -r ${CONFIG_DIR}/requirements.txt

USER 1001

CMD ["deploy.py"]

ENTRYPOINT ["python3"]
