## Application to demonstrate image search

# Usage: 
A dockerfile is provided. All that is required is a conda environment with Flask installed

There are a few fields that need to be populated in attalos_demo_app.py in order to work properly. These are located in the dictionaries at the top of the file

1. word_vec_file: The word vector file. This might be your original pretrained model or the updated word vectors if you are doing the joint optimization.
2. hdf5: The hdf5 file that is the output of the inference command. 
```bash
# Sample bash infer command
PYTHONPATH=$PWD python2 attalos/imgtxt_algorithms/infer.py    \
  /path/to/dataset.hdf5    \
  /path/to/dataset.json.gz \
  /path/to/output_file.hdf5 \
  /path/to/wordvector/file.gz \
  --word_vector_type=glove --batch_size=1024 --model_type=negsampling  \
  --hidden_units=2048,1024 \
  --model_input_path=/path/to/model_save_dir
```
3. txt in id_lookup is a tab delimited file where the 5th column has the full URL of the images. This assumes the dataset keys are the filenames from the url
```bash
# From the top level of the attlos repository

PYTHONPATH=$PWD/ FLASK_APP=attalos/imgtxt_algorithms/demo_app/attalos_demo_app.py flask run --host=0.0.0.0
```
