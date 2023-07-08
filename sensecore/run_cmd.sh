# standalone

composer run.py \
    --config-path yamls/hydra-yamls \
    --config-name SD-2-base-256-pexels.yaml \
    dataset.train_dataset.shuffle=false \
    batch_size=512

# multiple node
env

echo "WORLD_SIZE: ${WORLD_SIZE} RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR} MASTER_PORT: ${MASTER_PORT}"

export HYDRA_FULL_ERROR=1
export PROCESS_SIZE=$((WORLD_SIZE * 8))

cd /
rm /app/diffusion -rf
cp /mnt/CV_teamz/users/zhongyi/workspace/diffusion /app/diffusion -r
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
    --config-name SD-2-base-256-mixdata.yaml \
    batch_size=4096

