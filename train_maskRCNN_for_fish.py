import os
import json
import numpy as np
import skimage.draw
from mrcnn import utils
from PIL import Image
from shapely.geometry import Polygon
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

ROOT_DIR_0 = os.path.abspath("C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/")
ROOT_DIR = os.path.abspath("C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/maskrcnn/")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_OUT_DIR = os.path.join(ROOT_DIR_0, "processed_videos")
VIDEO_IN_DIR = os.path.join(ROOT_DIR_0, "videos")


class FishDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.num_train = 0
    def load_fish_dataset(self, dataset_dir, subsetsubfolder, split_ratio=0.8, is_training=True):
        self.add_class("FishNotFish", 1, "fish")
        self.add_class("FishNotFish", 2, "other")
        self.add_class("FishNotFish", 3, "null")
        data_dir = os.path.join(dataset_dir, subsetsubfolder)
        files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f != 'schema.json']
        self.num_train = int(len(files) * split_ratio)
        for i, json_file in enumerate(files):
            try:
                with open(os.path.join(data_dir, json_file)) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading JSON file {json_file}: {str(e)}")
                continue
            image_path = os.path.join(data_dir, data['image_filename'])
            image_id = os.path.splitext(json_file)[0]
            try:
                image = Image.open(image_path)
                width, height = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                continue
            if is_training and i < self.num_train:
                split = "train"
            elif not is_training and i >= self.num_train:
                split = "test"
            else:
                continue
            class_name = [label['label_class'] if label['label_class'] is not None else "null" for label in data['labels']]
            self.add_image(
                "FishNotFish",
                classes=class_name,
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                polygons=self.parse_polygons(data['labels']),
                split=split
            )
    def parse_polygons(self, labels):
        polygons = []
        for label in labels:
            class_name = label.get('label_class')
            regions = label.get('regions', [])
            for region in regions:
                polygon = [(int(pt['x']), int(pt['y'])) for pt in region]
                polygons.append(polygon)
        return polygons
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        masks = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)
        class_ids = [self.class_names.index(cls) for cls in image_info["classes"] if cls in self.class_names]
        for i, polygon in enumerate(image_info["polygons"]):
            cc, rr = skimage.draw.polygon(np.array(polygon)[:, 0], np.array(polygon)[:, 1])
            rr = np.clip(rr, 0, image_info["height"] - 1)
            cc = np.clip(cc, 0, image_info["width"] - 1)
            masks[rr, cc, i] = 1
        masks = masks.astype(np.uint8)
        return masks, np.array(class_ids)

dataset_dir = "C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/stills"
subsetsubfolder = ""
dataset = FishDataset()
dataset.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8)
dataset.prepare()

dataset_train = FishDataset()
dataset_train.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8)
dataset_train.prepare()

dataset_test = FishDataset()
dataset_test.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8, is_training=False)
dataset_test.prepare()

train_image_id = 0
train_image_info = dataset_train.image_info[train_image_id]
print("Training Image ID: ", train_image_info["id"], "\nPolygons: ", train_image_info["polygons"])

for i in range(len(dataset_train.image_info)):
    train_image_info = dataset_train.image_info[i]
    print("Training Image ID: ", train_image_info["id"], "\nPolygons: ", train_image_info["polygons"])

for i in range(len(dataset_train.image_info), len(dataset_train.image_info) + len(dataset_test.image_info)):
    test_image_info = dataset_test.image_info[i - len(dataset_train.image_info)]
    print("Testing Image ID: ", test_image_info["id"], "\nPolygons: ", test_image_info["polygons"])

train_image_id = 0
plt.imshow(dataset_train.load_mask(train_image_id)[0][:, :, 0], cmap='gray'), plt.show()

test_image_id = 0
plt.imshow(dataset_test.load_mask(test_image_id)[0][:, :, 0], cmap='gray'), plt.show()

def get_unique_classes(dataset_dir, subsetsubfolder):
    data_dir = os.path.join(dataset_dir, subsetsubfolder)
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f != 'schema.json']
    unique_classes = set()
    for json_file in json_files:
        with open(os.path.join(data_dir, json_file)) as f:
            data = json.load(f)
        labels = data.get('labels', [])
        for label in labels:
            class_name = label.get('label_class')
            if class_name:
                unique_classes.add(class_name.lower())
    return unique_classes

nclasses = len(get_unique_classes(dataset_dir, subsetsubfolder))

class TrainingConfig(Config):
    NAME = "Training"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2 + nclasses
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1280
    RPN_ANCHOR_SCALES = (40, 80, 160, 320, 640)
    TRAIN_ROIS_PER_IMAGE = 25
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5

config = TrainingConfig()
config.display()

image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "coco"
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

model_path = os.path.join(MODEL_DIR, "mask_rcnn_fish.h5")
model.keras_model.save_weights(model_path)

class InferenceConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = os.path.join(MODEL_DIR, "mask_rcnn_fish.h5")
model.load_weights(model_path, by_name=True)

image_id = np.random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config,image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, r['scores'], ax=get_ax())

image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
for image_id in image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    results = model.detect([image], verbose=0)
    r = results[0]
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
