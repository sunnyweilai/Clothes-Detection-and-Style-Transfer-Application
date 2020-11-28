
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# register new dataset to the detectron2
from cutomizeDataSet import get_clothes_dicts
json_dir = "/Users/sunny/Documents/OD_Dataset/CustomizedDataSet/test/annos"

for d in ["train_test"]:
    DatasetCatalog.register("clothes_" + d, lambda d = d: get_clothes_dicts(json_dir))
    MetadataCatalog.get("clothes_" + d).set(thing_classes=["short_sleeved_shirt", "long_sleeved_shirt",
                                                           "short_sleeved_outwear", "long_sleeved_outwear",
                                                           "vest", "sling", "shorts", "trousers", "skirt",
                                                           "short_sleeved_dress", "long_sleeved_dress",
                                                           "vest_dress", "sling_dress"])





# # visualize to verify register dataset
# clothes_metadata = MetadataCatalog.get("clothes_train_test")
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

# start training
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("clothes_train_test",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
#
#
# # Inference should use the config with parameters that are used in training
# # cfg now already contains everything we've set previously. We changed it a little bit for inference:
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)
#
# # visualize to verify register dataset
# from detectron2.utils.visualizer import ColorMode
# val_dataset = "/Users/sunny/Documents/OD_Dataset/CustomizedDataSet/test/annos"
# dataset_dicts = get_clothes_dicts(val_dataset)
# clothes_metadata = MetadataCatalog.get("clothes_train_test")
# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     output = predictor(img)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=clothes_metadata, scale=0.5, instance_mode= ColorMode.IMAGE_BW)
#     out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
#     print(output["instances"].to("cpu"))
#     # cv2.namedWindow("enhanced", 0);
#     # cv2.resizeWindow("enhanced", 2000, 800);
#     # cv2.imshow("enhanced", out.get_image()[:, :, ::-1])
#     # cv2.imwrite('validation.png', out.get_image()[:, :, ::-1])
#     # cv2.waitKey()







