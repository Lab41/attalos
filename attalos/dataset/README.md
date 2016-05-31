# Dataset download and ingestion

## MS COCO

## IAPR-TC12

## Visual Genome

This code will build a dataset preparer for the Visual Genome dataset, where you can get images and metadata by image key. Initialization will download the data by itself. However, it uses an API from the Visual Genome organizers. Since their API unzips the files, be prepared to have the same procedure done to this data. 

### Procedure for usage:
1. Download/clone the API directory from: https://github.com/ranjaykrishna/visual_genome_python_driver

2. You will need to add this API to your path. For example, 

   > sys.path.append( /my/api )

3. In that directory under 'src', soft-link the directory to the Visual Genome data. For example, if you downloaded the data into /my/dir/vg_data, and the api directory was downloaded to /my/api, then you'll need to do:

   > ln -s /my/dir/vg_data /my/api/src/data





