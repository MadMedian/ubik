docker run -w /batch-model/ --entrypoint python --rm --name simple-batch-model simple-batch-model:latest \
  -m unittest /batch-model/src/tests/test_model.py