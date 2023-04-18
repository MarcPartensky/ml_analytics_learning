FROM python:3.9.16

RUN pip install keras tensorflow kaggle numpy jupyterlab pillow
WORKDIR /root/tsi-ml

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0"]
