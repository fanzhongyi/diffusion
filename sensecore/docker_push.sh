docker build -t zhongyi/mosaicmlsd:test .

docker tag zhongyi/mosaicmlsd:test registry.cn-sh-01.sensecore.cn/zteam-ccr/zhongyi_mosaicmlsd1:test

docker push registry.cn-sh-01.sensecore.cn/zteam-ccr/zhongyi_mosaicmlsd1:test

# docker run -it --rm --ipc host --net host --gpus all --name zhongyi-mosaicml -v /mnt:/mnt -v /mnt/CV_teamz/users/zhongyi/workspace/diffusion:/app/diffusion zhongyi/mosaicmlsd:test bash
