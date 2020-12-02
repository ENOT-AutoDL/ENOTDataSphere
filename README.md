# ENOTDataSphere
This is a repository with ENOT framework usage examples in Yandex DataSphere.

### Install
* Go to "install" folder;
* Open Install.ipynb ;
* Execute all the cells one by one;
* Now you can use all Jupyter Notebooks from this repository;


### To reproduce results from webinar:
* Create your own project in Yandex DataSphere - https://cloud.yandex.ru/services/datasphere 
* Clone this repository into your DataSphere project (Git -> Clone);
* Download models from the link below, unzip file(Snippets -> Extract ZIP file.py) \
and put it in the root directory of the project(so path to models should be '/home/jupyter/work/resources/ENOTDataSphere/models/'): 
https://yadi.sk/d/nbaV1N1tQMSPpg
* Download test dataset from the link below, unzip it(Snippets -> Extract ZIP file.py) \
and put "data" folder into "mmdet_tools" folder(so path to your data should be '/home/jupyter/work/resources/ENOTDataSphere/mmdet_tools/data/'):
https://yadi.sk/d/MwN9o5LmLi5Cvg
* Download test video from this link, create folder "video" and put downloaded video there:
https://yadi.sk/i/hE05IF9-OEwvKg


### Train custom dataset with MMDET+ENOT
##### To train your custom dataset for detection and use optimization framework ENOT you should:
###### 1 - Prepaire your dataset to COCO annotation format. 
* You should have 3 .json files, like *. train.json, test.json, val.json .* and 3 folders(train, test, val) with images. Make folder(like 'my_dataset_name') and copy all these files into it. About COCO annotation format you can read here - https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html
* In 'mmdet_tools' make folder 'data' and copy your directory  with dataset('my_dataset_name') to 'data'.
###### 2 - Prepaire configs for ENOT_Pretrain, ENOT_search and tune phases.
* In 'mmdet_tools/configs/_base_/datasets' in files 'mask_face.py' and 'mask_face_pretrain.py' set to variable 'data_root' path to your dataset.
Set 'ann_file' parameter for train, test and val dictionaries with paths to yours .json files(train.json, test.json, val.json)
* If you want to change searchspace you should do it in 'mmdet_tools/configs/_base_/models/search_space_ssd_masks.py'
* For enot_pretrain phase you should change paths to your dataset in 'mmdet_tools/configs/wider_face/search_space_ssd_masks.py', also you can set here augmentations you want.
* For enot_search phase you should change paths to your dataset in 'mmdet_tools/configs/wider_face/search_space_ssd_masks_search.py', dataset and augmentations should be the same like in pretrain phase!
####### Detailed information about MMDetection config system you can find here - https://mmdetection.readthedocs.io/en/latest/tutorials/config.html



#### Tutorials for ENOT framework you can find here:
https://github.com/ENOT-AutoDL/ENOT_Tutorials




