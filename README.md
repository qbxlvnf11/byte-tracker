
Description
=============

#### - ByteTrack

- ByteTrack
  - Effective and generic association method, tracking by associating almost every detection box instead of only the high score ones
- BYTE
  - First Association
    - Matcing high score detection bounding boxes based on motion similarity and appearance similarity
    - Next, Kalman filter is applied to predict the position of the tracklets of the next frame
    - This process is similar to traditional trackers
  - Second Association
    - The difference of BYTE is considering the low score detection bounding boxes once more
    - Based on the same motion similarity, we match unmatched tracklets and low score boxes such as the occlusion boxes
- More details
  - https://blog.naver.com/qbxlvnf11/222784457900
  - [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864.pdf)
- Pseudo-code of BYTE

<img src="" width="35%"></img>

#### - Yolov3
- You only look once (YOLO) is one of the the powerful and real-time 1-stage object detection systems
- Improved features compared to yolov2: FPN,shortcut connection, logistic regression etc.
- More details: [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
  
Contents
=============

#### - ByteTrack
- Identifying objects detected by yolov3

#### - Yolov3 Train/inference
- Train yolov3 model
- Detect image

#### - Yolov3 TensorRT Engine
- Convert yolov3 Pytorch weigths to TensorRT engine
- Real-time inference with yolov3 TensorRT engine

#### - Config files
- byte_tracker_config.ini: byte tracker parameters
- yolov3_config.ini: yolov3 model parameters
- train_config.ini: yolov3 train parameters
- tensorrt_config.ini: yolov3 tensorrt parameters

Yolov3 Run Environments with TensorRT 7.2.2 & Pytorch
=============

#### - Docker with TensorRT
- https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_20-12.html#rel_20-12

#### - Docker pull
```
docker pull qbxlvnf11docker/byte_tracker_yolov3:latest
```

#### - Docker run
```
nvidia-docker run -it --name byte_tracker_yolov3 -v {yolo-v3-tensorrt-repository-path}:/workspace/Byte-Tracker-Yolov3 -w /workspace/Byte-Tracker-Yolov3 qbxlvnf11docker/byte_tracker_yolov3:latest bash
```

How to use
=============

#### - Detecting Image with Yolov3 and Multi-Objects Tracking with ByteTrack
- Params: refer to config files and parse_args()
```
python main.py --mode yolov3-detection-img
```

#### - Build Yolov3 def cfg
```
./create_model_def.sh {class_num} {cfg_name}
```

#### - Download Pretrained Yolov3 Weights
```
./download_weights.sh
```

#### - Train Yolov3 Model
- Params: refer to config files and parse_args()
```
python train.py --mode yolov3-train
```

#### - Build TensorRT Engine
- Params: refer to config files and parse_args()
```
python yolov3_convert_onnx_tensorrt.py --yolov3_config_file_path ./config/yolov3_config.ini --tensorrt_config_file_path ./config/tensorrt_config.ini
```

Dataset
=============

#### Multi-Objects Tracking Test Dataset

- GOT-10k: http://got-10k.aitestunion.com/

#### Detection Train & Test Dataset

- Download COCO2014 dataset
```
./get_coco_dataset.sh
```

#### - Build Data Json Files for Train Yolov3
- Building data json for optimizing yolov3
- In train process, read builded data json file and get train data
- Params: refer to parse_args()
```
python yolov3_convert_onnx_tensorrt.py --target coco2014 --data_folder_path ./data/train_data/coco --save_folder_path ./data/data_json/coco
```

#### - Format of Data Json Files
- parsing_data_dic['class_format'] = type of class ('name' or 'id')
- parsing_data_dic['label_scale'] = scale of label ('absolute' or 'relative')
- parsing_data_dic['image_list'] = [{'id'-image id, 'image_file_path'-image file path}, ...]
- parsing_data_dic['object_boxes_list'] = [{'image_id'-image id, 'object_box_num'-number of the object per image, 'object_box_id_list'-[object box id, ...], 'object_name_list'-[object name, ...], 'object_box_list'-[[center x, center y, box_width, box_height], ...], 'object_box_size_list'-[object box size, ...], }, ...]
- parsing_data_dic['image_num'] = number of the image
- parsing_data_dic['object_boxes_num'] = [number of the total objects, ...]

References
=============

#### - ByteTrack Paper
```
@article{ByteTrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Yifu Zhang et al.},
  journal = {arXiv},
  year={2021}
}
```

#### - Yolov3 Paper
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

#### - ByteTrack Pytorch

https://github.com/ifzhang/ByteTrack

#### - Yolov3 with TensorRT

https://github.com/qbxlvnf11/yolo-v3-tensorrt

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

