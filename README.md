<h2><p align="center">Efficient Stable Diffusion Training with Composer</p></h2>

**WIP**

checkout **model-3B** branch for another version, corresponding config file is
`SD-3B-base-[256]-[pexels].yaml` (deprecated)

checkout **stable** branch for sd2.1 version, corresponding config file is
`SD-2-base-[256, 512]-[wds, pexels, mixdata].yaml` (WIP)

**main** : not a stable branch.

# Training from stratch

## Single Nodes (for ipdb debug)

1. build docker images (template in `./sensecore/Dockerfile`);
2. create docker container and mount workspace to `/app/diffusion` (can be modified by yourself);
3. start docker container;
4. run debug cmd:

```bash
CUDA_VISIBLE_DEVICES='6' HYDRA_FULL_ERROR=1 composer run.py --config-path yamls/hydra-yamls --config-name SD-3B-base-256-pexels.yaml batch_size=2 dataset.train_dataset.num_workers=1 dataset.eval_dataset.num_workers=0
```

or

```bash
CUDA_VISIBLE_DEVICES='3' HYDRA_FULL_ERROR=1 python run.py --config-path yamls/hydra-yamls --config-name SD-3B-base-256-pexels.yaml batch_size=2 dataset.train_dataset.num_workers=1 dataset.eval_dataset.num_workers=0
```


## Multiple Nodes
1. create docker container and mount filesystem.
2. submit job

```bash
env

echo "WORLD_SIZE: ${WORLD_SIZE} RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR} MASTER_PORT: ${MASTER_PORT}"

export HYDRA_FULL_ERROR=1
export PROCESS_SIZE=$((WORLD_SIZE * 8))

cd /
rm /app/diffusion -rf
cp /path/to/diffusion /app/diffusion -r
cd /app/diffusion

composer \
    --master_port ${MASTER_PORT} \
    --master_addr ${MASTER_ADDR} \
    --world_size ${PROCESS_SIZE} \
    --node_rank ${RANK} \
    --stdout '{world_size}-{rank}.out' \
    --stderr '{world_size}-{rank}.err' \
run.py \
    --config-path yamls/hydra-yamls \
    --config-name SD-3B-base-256-pexels.yaml \
    batch_size=1024
```
