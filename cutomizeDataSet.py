# from detectron2.data.datasets import register_coco_instances
# # register_coco_instances("my_dataset_train", {}, "customizedDataSet/validation/annos", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/annos/000001.json", "/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/image")

import json

# directory = os.fsencode("/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/annos")
#
# path1 = ("/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/annos/000001.json")
# path2 = ("/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/annos/000002.json")
# with open(path1) as f1:
#   item1 = json.load(f1)
# with open(path2) as f2:
#   item2 = json.load(f2)
#
# data = dict(item1, **item2)
# merged_data = json.dumps(data)
# print(merged_data)

from detectron2.structures import BoxMode
import os
import cv2

json_dir = "/Users/sunny/Documents/ObjectDetection/CustomizedDataSet/validation/annos"
def get_clothes_dicts(json_dir):
    dataset_dicts = []
    for annos in os.listdir(json_dir):
        json_path = json_dir + '/%s'%(annos)
        with open(json_path) as f:
            img_anns = json.load(f)

        objs = []
        for _,anno in img_anns.items():
            bbox = anno['bounding_box']
            category_id = anno['category_id']
            keypoints = anno['landmarks']
            obj = {

            }

        img_anns.update({'filename':'%s.jpg'%(anno[:-5])})
        img_anns.update({'image_id':'%s'%(anno[:-5])})







get_clothes_dicts(json_dir)






# def get_each_json_dicts(json_file, img_file):
#     with open(json_file) as f:
#         img_anns = json.load(f)
#
#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
#
#         filename = os.path.join(img_dir, v["filename"])
#         height,width = cv2.imread(filename).shape[:2]
#
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#
#         annos = v["regions"]