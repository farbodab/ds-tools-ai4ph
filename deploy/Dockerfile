FROM python:3.8.3

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "80"]
