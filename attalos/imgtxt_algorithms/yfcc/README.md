# YFCC Data Processing

The YFCC 100million dataset's size requires some special processing. The starting assumption is that you've downloaded the metadata files describing the colleciton located at https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67
You will need two sets of files:
*  yfcc100m_dataset-X.bz2: Files containing the metadata for the yfcc collection (X is a number from 0 through 9)
*  yfcc100m_hash.bz2: Maps the hash values used in the files below to the Ids in the metadata

This also assumes that you've downloaded the VGG features from (http://www.multimediacommons.org/getting-started) located at: http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=features/image/vgg-vlad-yfcc/vgg/
The files are named "yfcc100m_dataset-X.tar.gz" where X is a number from 0 through 9


Preparing the curated data is a multistep processing:

1. Create the json file normally created by the attalos.preprocessing.text script

2. Create the hdf5 file normally created by the attalos.preprocessing.image scripts

3. (Optional) Create a custom w2v model based on the image tags

4. Create the curated file using the create_curated.py file which takes a dataset iterator (requiring steps 1 and 2)

5. Run actual negative sampling code

### Example Usage
```bash
# Step 1
python create_yfcc_text_json.py /path/to/yfcc_100m_hash.bz2 /path/to/yfcc100m_dataset-X.bz2

# Step 2
# Input take from http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=features/image/vgg-vlad-yfcc/vgg/
python create_hdf5_file.py /path/to/yfcc100m_dataset-X.tar

# Step 3
python create_w2v_model.py /path/to/yfcc_metadata /path/to/my_w2v_model.gz

# Step 4
PYTHONPATH=/path/to/attalos:$PYTHONPATH python create_curated.py /path/to/yfcc100m_dataset-X.tar.hdf5 /path/to/yfcc100m_dataset-X.bz2.json.gz /path/to/yfcc100m_curated_dataset-X.tar.hdf5  /path/to/my_w2v_model.gz

# Step 5
PYTHONPATH=/path/to/attalos:$PYTHONPATH python negsampling_regression.py /path/to/yfcc100m_curated_dataset-X.tar.hdf5
```
