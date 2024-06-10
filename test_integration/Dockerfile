FROM python:3.9
RUN apt-get update -y
RUN apt-get -y install default-mysql-client
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

CMD ["/bin/bash"]