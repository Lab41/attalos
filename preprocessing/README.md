# MSCOCO

# Assumptions:
1. Make has already been run creating the docker containers
2. The MSCOCO data (http://msvocds.blob.core.windows.net/coco2014/train2014.zip) has been downloaded
  * The data should be unzipped such that your folder structure looks like /path/to/data/train2014



Generate Inception v3 Features
```
# No GPU
docker run -it\
    --volume /path/to/data:/local_data \
    --volume $PWD:/image_extract \
    lab41/l41-caffe-keras-tf /bin/bash

# With GPU Support
docker run -it\
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia0:/dev/nvidia0 \
    --volume /path/to/data:/local_data \
    --volume $PWD:/image_extract \
    lab41/l41-caffe-keras-tf /bin/bash

# Run script creating image features
# Suppress deprecation warning since it occurs once per image per batch normalization in the inception graph (aka a lot)
python /image_extract/extract_inception_features.py --image_dir /local_data/train2014 --output_fname /local_data/mscoco_train2014.hdf5 | grep -v "deprecated"
```
