# MSCOCO

# Assumptions:
1. Make has already been run creating the docker containers
2. Assumes your data is in /local_data/{dataset_name}


# Start relevant docker container
```
# No GPU
docker run -it\
    --volume /path/to/data:/local_data \
    --volume /path/to/attalos:/attalos \
    lab41/l41-caffe-keras-tf /bin/bash

# With GPU Support
docker run -it\
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia0:/dev/nvidia0 \
    --volume /path/to/data:/local_data \
    --volume /path/to/attalos:/attalos \
    lab41/l41-caffe-keras-tf /bin/bash

```
# Simplified Steps
```
cd /attalos/preprocessing/

./process_data.sh /path/to/data/directory mscoco
# Repeat for other desired datasets (i.e. espgame, iaprtc, visualgenome)

```

# Run Extract Scripts in Docker container
```

cd /attalos

#Extract Text Features
PYTHON_PATH=$PYTHON_PATH:/attalos python attalos/preprocessing/text/extract_text_features.py \
    --dataset_dir /local_data/mscoco \
    --output_fname /local_data/features/text/mscoco_train2014_text.json.gz \
    --dataset_type mscoco \
    --split train

#Extract Inception Features
# Suppress deprecation warning since it occurs once per image per batch normalization in the inception graph (aka a lot)
PYTHON_PATH=$PYTHON_PATH:/attalos python attalos/preprocessing/image/extract_inception_features.py \
    --dataset_dir /local_data/mscoco \
    --output_fname /local_data/features/image/mscoco_train2014_inception.hdf5 \
    --dataset_type mscoco \
    --split train 2>&1 | grep -v "deprecated"
```
