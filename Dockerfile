FROM python:3
COPY . /cloud
WORKDIR /cloud
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["iWebLens_server.py"]

