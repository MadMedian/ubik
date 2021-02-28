# A simple batch model
## Overview
The container has the following structure:
```
/batch-model/src
                |-- model.py            # contains the model code
                |-- train.py            # entry point for training the model
                |-- predict.py          # entry point for getting predictions
                |-- data                # where we will mount the outside data folder
                '-- tests
                    '-- test_model.py   # unit tests, no data files needed
```

## Instructions
This example is based on the python slim image:
```shell script
docker pull python:3.8.8-slim
```

Build the docker image:
```shell script
sh build_docker.sh
```

Run the unit tests:
```shell script
sh run_unittests.sh
```

For training and prediction we mount a local directory containing the data into `/batch-model/src/data`.
To simplify, we agree to use the following directory structure:
```
/batch-model/src/data
                    |-- train_input
                    |   |-- X.npy               # train input data
                    |   '-- y.npy               # train input label
                    |
                    |-- model
                    |   '-- model.joblib        # the model will be saved here
                    |
                    |-- pred_input
                    |   '-- X_batch.npy         # prediction input data
                    |
                    '-- pred_output
                        '-- y_pred_batch.npy    # prediction output data
```
This convention can easily be altered in the `train.py` and `predict.py` files. 
One can also change the type or number of the data files (dataframes, CSVs, ...).

Run the model training:
```shell script
sh run_train.sh /../path_to_local_directory/
```

Run the batch prediction - the model has to be trained before and saved in the relevant subfolder:
```shell script
sh run_batch_prediction.sh /../path_to_local_directory/
```