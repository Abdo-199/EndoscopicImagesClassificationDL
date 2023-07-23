FROM python:3.10.11

COPY requirements.txt .

RUN python -m pip cache purge

RUN pip install --upgrade -r requirements.txt

COPY server server/

EXPOSE 5000

CMD ["python", "server/main.py"]
