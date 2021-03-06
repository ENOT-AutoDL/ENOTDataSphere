# ENOTDataSphere
This is a repository with ENOT framework usage examples in Yandex DataSphere.

## Install
* Go to "install" folder;
* Open Install.ipynb ;
* Execute all the cells one by one;
* Now you can use all Jupyter Notebooks from this repository;


## To reproduce results from the webinar:
* Create your own project in Yandex DataSphere - https://cloud.yandex.ru/services/datasphere 
* Clone this repository into your DataSphere project (Git -> Clone);
* Download models from the link below, unzip file(Snippets -> Extract ZIP file.py) \
and put it in the root directory of the project(so the path to models should be './ENOTDataSphere/models/base_model.pth'): 
https://yadi.sk/d/nbaV1N1tQMSPpg
* Download test dataset from the link below, unzip it(Snippets -> Extract ZIP file.py) \
and create "data" folder in "mmdet_tools" folder and copy unziped folder('masks') in './ENOTDataSphere/mmdet_tools/data/'(so the path to your data should be './ENOTDataSphere/mmdet_tools/data/'):
https://yadi.sk/d/MwN9o5LmLi5Cvg
* Download test video from this link, create folder "video" and put downloaded video there:
https://yadi.sk/i/hE05IF9-OEwvKg

##### Now you can use all notebooks in this repository
* demo_video.ipynb - demo of masks detection on video. You can download your own video, change path to the video and detect masks;
* pretrain_search.ipynb - in this notebook you can find examples of enot_pretrain, enot_search phases and get test metrics on the dataset;
* train_baseline.ipynb - example of train baseline model with MMDetection;


## Train detector on custom dataset with MMDET+ENOT
### To train your custom dataset for detection and use optimization framework ENOT you should:
#### 1 - Prepare your dataset to COCO annotation format. 
* You should have 3 .json files, like *. train.json, test.json, val.json .* and 3 folders(train, test, val) with images. Make a folder(like 'my_dataset_name') and copy all these files into it. About COCO annotation format you can read here - https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html
* In 'mmdet_tools' make a folder 'data' and copy your directory  with dataset('my_dataset_name') to 'data'.
#### 2 - Prepare configs for ENOT_Pretrain, ENOT_search and tune phases.
* In 'mmdet_tools/configs/_base_/datasets' in files 'mask_face.py' and 'mask_face_pretrain.py' set to variable 'data_root' path to your dataset.
Set 'ann_file' parameter for train, test and val dictionaries with paths to yours .json files(train.json, test.json, val.json)
* If you want to change searchspace you should do it in 'mmdet_tools/configs/_base_/models/search_space_ssd_masks.py'
* For enot_pretrain phase you should change paths to your dataset in 'mmdet_tools/configs/wider_face/search_space_ssd_masks.py', also you can set here augmentations you want.
* For enot_search phase you should change paths to your dataset in 'mmdet_tools/configs/wider_face/search_space_ssd_masks_search.py', dataset and augmentations should be the same like in pretrain phase!
###### Detailed information about MMDetection config system you can find here - https://mmdetection.readthedocs.io/en/latest/tutorials/config.html

#### 3 - run enot_pretrain, enot_search phases
To start enot_pretrain you should:
* Prepare config(see 2);
* In 'mmdet_tools/run_enot_pretrain.py' in pretrain_cfg dictionary you should change: 'epochs' - number of epochs, 'mmdet_config_path' - path to 'mmdet_tools/configs/_base_/models/search_space_ssd_masks.py', 'experiment_dir' - path to save checkpoints. In 'enot_pretrain' function you can change type of optimizer, learning rate, set scheduler, and batch size;
* When all configs, learning procedure and paths in 'mmdet_tools/run_enot_pretrain.py' are ready, from jupyter notebook just call 'run_enot_pretrain' function from 'mmdet_tools/run_enot_pretrain.py';
* Prepare config(see 2);
* Choose best checkpoint from pretrain phase;
* In 'mmdet_tools/run_enot_search.py' in pretrain_cfg dictionary you should change: 'epochs' - number of epochs, 'mmdet_config_path' - path to 'mmdet_tools/configs/_base_/models/search_space_ssd_masks.py', 'experiment_dir' - path to save checkpoints, 'pretrain_checkpoint_path' - path to best checkpoint from pretrain phase. In 'enot_search' function you can change type of optimizer, learning rate, set scheduler, and batch size;
* When all configs, learning procedure and paths in 'mmdet_tools/run_enot_search.py' are ready, from jupyter notebook just call 'run_enot_search' function from 'mmdet_tools/run_enot_search.py'. If you can set up parameter 'latency_loss_weight' to vary complexity of model to find, bigger 'latency_loss_weight' - more lightweight model you will find;

#### 4 - run tune found model
* Prepare config 'mmdet_tools/configs/_base_/models/mobilenet_from_search_space_mask.py', set indices of best models from search phase. And 'mmdet_tools/configs/wider_face/mobilenet_from_search_space_mask.py' here you should set path to dataset, augmentations, optimizer, learning rate;
* Export weights(.pth file) of the best model. Example you can see in 'mmdet_tools/build_searched_arch_ckpt.py';
* Call train from 'mmdet_tools/tools/train_baseline.py'. Set parameter 'config' - path to 'mmdet_tools/configs/wider_face/mobilenet_from_search_space_mask.py', 
'full_pretrained_model' - path to exported .pth file;


## Tutorials for ENOT framework you can find here:
https://github.com/ENOT-AutoDL/ENOT_Tutorials




