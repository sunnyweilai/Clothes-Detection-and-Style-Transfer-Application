from detectron2.data.datasets import register_coco_instances
import json
import os


from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
import random


json_dir = "/Users/sunny/Documents/OD_Dataset/nuts/trainval.json"
image_dir = "/Users/sunny/Documents/OD_Dataset/nuts/images"
register_coco_instances("fruits_nuts", {}, json_dir, image_dir)

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
print(fruits_nuts_metadata)

# dataset_dicts = DatasetCatalog.get("fruits_nuts")
# for d in random.sample(dataset_dicts, 2):
#     img = cv2.imread(d["file_name"])
#
#     visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.namedWindow("enhanced", 0);
#     cv2.resizeWindow("enhanced", 2000, 800);
#     cv2.imshow("enhanced", out.get_image()[:, :, ::-1])
#     cv2.imwrite('verify1.png', out.get_image()[:, :, ::-1])
#     cv2.waitKey()
