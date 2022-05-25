docker run --rm --gpus device=0 -it -v $PWD:/app \
--network=host \
--shm-size=32G \
watermark-detector:v0 bash