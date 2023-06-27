composer run.py \
    --config-path yamls/hydra-yamls \
    --config-name SD-2-base-256-pexels.yaml \
    dataset.train_dataset.shuffle=false \
    batch_size=512
