import json
import os
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import random


json_dir = "/Users/sunny/Documents/OD_Dataset/CustomizedDataSet/test/annos"
image_dir = "/Users/sunny/Documents/OD_Dataset/CustomizedDataSet/test/image"

def get_clothes_dicts(json_dir):
    dataset_dicts = []
    for annos in os.listdir(json_dir):
        if 'json' in annos:
            json_path = json_dir + '/%s'%(annos)
            with open(json_path) as f:
                img_anns = json.load(f)

            record = {}
            # add general info
            image_filename = image_dir + '/%s.jpg'%(annos[:-5])
            record["file_name"] = image_filename
            record["image_id"] = '%s'%(annos[:-5])

            height, width = cv2.imread(image_filename).shape[:2]
            record["height"] = height
            record["width"] = width

            objs = []
            for key,anno in img_anns.items():
                if "item" in key:
                    bbox = anno['bounding_box']
                    segmentation = anno['segmentation']

                    # the index starts from 0
                    category_id = anno['category_id'] - 1


                    obj = {
                        "bbox":bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation":segmentation,
                        "category_id":category_id,

                    }
                    objs.append(obj)
                record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

# dataset = get_clothes_dicts(json_dir)
#
# json_name = 'clothes_dataset_test.json'
# with open(json_name, 'w') as f:
#   json.dump(dataset, f)

# register new dataset to the detectron2
# for d in ["train", "val"]:
#     DatasetCatalog.register("clothes_" + d, lambda d = d: get_clothes_dicts(json_dir))
#     MetadataCatalog.get("clothes_" + d).set(thing_classes=["short_sleeved_shirt", "long_sleeved_shirt",
#                                                            "short_sleeved_outwear", "long_sleeved_outwear",
#                                                            "vest", "sling", "shorts", "trousers", "skirt",
#                                                            "short_sleeved_dress", "long_sleeved_dress",
#                                                            "vest_dress", "sling_dress"])
# clothes_metadata = MetadataCatalog.get("clothes_train")


# visualize to verify register dataset
# dataset_dicts = get_clothes_dicts(json_dir)
# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#
#     visualizer = Visualizer(img[:, :, ::-1], metadata=clothes_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.namedWindow("enhanced", 0);
#     cv2.resizeWindow("enhanced", 2000, 800);
#     cv2.imshow("enhanced", out.get_image()[:, :, ::-1])
#     cv2.imwrite('verify.png', out.get_image()[:, :, ::-1])
#     cv2.waitKey()

