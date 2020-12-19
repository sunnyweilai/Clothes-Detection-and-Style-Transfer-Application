
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer


def train_model(trainset_name: str,learning_rate: int, num_iteration: int, batch_per_image: int,num_classes: int):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (trainset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = learning_rate  # pick a good LR
    cfg.SOLVER.MAX_ITER = num_iteration    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_per_image   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor





# ---------- select a content image to get trained result_img
def get_content_output(content_img,trained_model):
    output = trained_model(content_img)
    return output



def visualize_content_output(content_img,trained_model,metadata):
    output = get_content_output(content_img,trained_model)
    v = Visualizer(content_img[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5
                   )
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    print(output["instances"].to("cpu"))
    cv2.namedWindow("enhanced", 0);
    cv2.resizeWindow("enhanced", 2000, 800);
    cv2.imshow("enhanced", out.get_image()[:, :, ::-1])
    cv2.imwrite('validation.png', out.get_image()[:, :, ::-1])
    cv2.waitKey()






# ------------------------------------get masked image for the selected image
import torch,torchvision
import math


def calculateDistance(x1, y1, x2, y2):
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist


def get_masked_content_image(output, content_image):
    pred_masks = output["instances"].get_fields()["pred_masks"]
    boxes = output["instances"].get_fields()["pred_boxes"]

    # get main mask if the area is >= the mean area of boxes and is closes to the centre
    masker = pred_masks[np.argmin([calculateDistance(x[0], x[1], int(content_image.shape[1] / 2), int(content_image.shape[0] / 2)) for i, x in
                                   enumerate(boxes.get_centers()) if
                                   (boxes[i].area() >= torch.mean(boxes.area()).to("cpu")).item()])].to(
        "cpu").numpy().astype(np.uint8)
    masked_image = cv2.bitwise_and(content_image, content_image, mask=masker)
    return masked_image,masker
