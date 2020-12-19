from PIL import Image
import numpy as np
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog

import clothes_data
import clothes_train
import style_transfer


# -------- get dataset dir
annos_dir = "/Users/sunny/Documents/OD_Dataset/CustomizedDataSet/train/annos"


# ----------- register dataset
for d in ["train", "val"]:
    DatasetCatalog.register("clothes_" + d, lambda d=d: clothes_data.get_clothes_dicts(annos_dir))
    MetadataCatalog.get("clothes_" + d).set(thing_classes=[ "short_sleeved_shirt", "long_sleeved_shirt","short_sleeved_outwear", "long_sleeved_outwear",
                                                           "vest", "sling", "shorts", "trousers", "skirt",
                                                           "short_sleeved_dress", "long_sleeved_dress",
                                                           "vest_dress", "sling_dress"])

# --------- train model
predictor = clothes_train.train_model("clothes_train", 0.00025, 2, 128, 13)

# ---------- select a content image
dataset_dicts = clothes_data.get_clothes_dicts(annos_dir)
clothes_metadata = MetadataCatalog.get("clothes_train")
content_image = cv2.imread(dataset_dicts[4]["file_name"])


# ---------- get trained content masked image and masker
output = clothes_train.get_content_output(content_image, predictor)
content_masked_image,masker = clothes_train.get_masked_content_image(output,content_image)


# ----------- start style transfer
content_img = Image.fromarray(content_masked_image, 'RGB')
style_img = Image.open("img/style_2.jpg")
best_style_img, best_loss = style_transfer.run_style_transfer(content_img, style_img, num_iterations=50)


# ----------- resize the image back to the original
def resizeOutputImage(input_image, result_array):
    result_img = Image.fromarray(result_array)
    scale_width = input_image.size[0]/result_img.size[0]
    scale_height = input_image.size[1]/result_img.size[1]
    recreate_img = result_img.resize((round(result_img.size[0]*scale_width), round(result_img.size[1]*scale_height)), Image.ANTIALIAS)
    recreate_arrary = np.asarray(recreate_img)
    return recreate_arrary


def get_style_masked_image(content_img, output_image, masker):
    recreate_output_array = resizeOutputImage(content_img, output_image)
    style_masked_image = cv2.bitwise_and(recreate_output_array, recreate_output_array, mask=masker)
    return style_masked_image


def get_final_img(content_img, output_masked_img):
    # get the output masked image
    style_masked_image = get_style_masked_image(content_img, output_masked_img)
    final_img = cv2.add(content_img, style_masked_image)
    return final_img


final_result = get_final_img(content_image, best_style_img)
