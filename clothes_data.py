import json
import os
from detectron2.structures import BoxMode
import cv2


def get_clothes_dicts(json_dir):
    image_dir = json_dir.replace("annos", "image")
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

