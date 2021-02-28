docker run --entrypoint python \
  -v "$1":/batch-model/src/data \
  --rm --name simple-batch-model simple-batch-model:latest predict.py
