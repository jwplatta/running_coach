FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 7860

ENV GRADIO_SERVER_NAME 0.0.0.0

CMD ["python", "app.py"]