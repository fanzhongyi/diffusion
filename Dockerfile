FROM mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04

ENV OMP_NUM_THREADS=20

RUN echo "tmpfs /dev/shm tmpfs defaults,size=4g 0 0" >> /etc/fstab \
    && apt-get update \
    && apt-get -y install pdsh vim \
    && rm -rf /var/lib/apt/lists/*

COPY petrel-oss-sdk-2.2.4 /app/petrel-oss
COPY diffusion /app/diffusion
RUN cd /app/petrel-oss \
    && python setup.py sdist \
    && pip install --no-cache-dir dist/* \
    && rm -rf dist \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e /app/diffusion

WORKDIR /app/diffusion
