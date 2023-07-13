FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV OMP_NUM_THREADS=20
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN echo "tmpfs /dev/shm tmpfs defaults,size=20g 0 0" >> /etc/fstab \
    && echo "tmpfs /mem_fs tmpfs defaults,size=30g 0 0" >> /etc/fstab \
    && apt-get update \
    && apt-get -y install pdsh vim graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/*

COPY petrel-oss-sdk-2.2.4 /app/petrel-oss
COPY diffusion /app/diffusion
COPY xformers /app/xformers

RUN pip install --upgrade pip \
    && cd /app/petrel-oss \
    && python setup.py sdist \
    && pip install --no-cache-dir dist/* \
    && rm -rf dist \
    && pip uninstall -y pillow && pip install pillow-simd \
    && pip install ipdb \
    && pip install -v -U /app/xformers \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e /app/diffusion

WORKDIR /app/diffusion
