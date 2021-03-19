#FROM tensorflow/tensorflow:latest-gpu
#FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
#FROM tensorflow/tensorflow:latest-gpu-py3
FROM python:3.7-slim AS builder
MAINTAINER ADAM GREEN


COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential \
    wget \
    python3-pip\
    git

RUN python -m venv /opt/venv

RUN pip3 install --upgrade pip

ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install -r requirements.txt

RUN echo 'test'
RUN git clone https://github.com/aasgreen/simple-scipy-seg.git /app/work

FROM python:3.7-slim as build-image
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/work /app/work

ENV PATH="/opt/venv/bin:$PATH"
ENV HOME /app/work
WORKDIR /app/work/


#CMD ["bash","call_defect.sh"]
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
#RUN export PATH=$PATH:/app/src/data && echo "export PATH=$PATH" >> /etc/bash.bashrc


CMD ["bash"]
#CMD ["uvicorn", "ml:app", "--host", "0.0.0.0", "--reload"]
#CMD ["python", "app.py"]
#CMD ["python"]
