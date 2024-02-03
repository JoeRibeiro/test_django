import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project 
ROOT_DIR = os.path.abspath("C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/maskrcnn/") # C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/maskrcnn/   frames output_masks

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")



import os
import json
import numpy as np
import skimage.draw
from mrcnn import utils
from PIL import Image  # Pillow library
from shapely.geometry import Polygon


class FishDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.num_train = 0  # Initialize as an instance variable
    def load_fish_dataset(self, dataset_dir, subsetsubfolder, split_ratio=0.8, is_training=True):
        # Add classes (adjust based on your dataset)
        self.add_class("FishNotFish", 1, "fish")
        self.add_class("FishNotFish", 2, "other")
        # Define data directory
        data_dir = os.path.join(dataset_dir, subsetsubfolder)
        # List all files in the data directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f != 'schema.json']
        # Calculate the number of files for training based on the split ratio
        self.num_train = int(len(files) * split_ratio)
        for i, json_file in enumerate(files):
            try:
                # Load JSON file
                with open(os.path.join(data_dir, json_file)) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading JSON file {json_file}: {str(e)}")
                continue
            # Extract image details
            image_path = os.path.join(data_dir, data['image_filename'])
            image_id = os.path.splitext(json_file)[0]  # Extract image ID from filename
            try:
                # Get image width and height
                image = Image.open(image_path)
                width, height = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                continue
            # Determine if the image is for training or testing
            if is_training and i < self.num_train:
                split = "train"
            elif not is_training and i >= self.num_train:
                split = "test"
            else:
                continue  # Skip this image if not in the desired split
            # Add image to dataset
            class_name = [label['label_class'] for label in data['labels']]
            self.add_image(
                "FishNotFish",
                classes = class_name,
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                polygons=self.parse_polygons(data['labels']),
                split=split  # Add split information to image metadata
            )
    def parse_polygons(self, labels):
        polygons = []
        for label in labels:
            class_name = label.get('label_class')  # Use get to handle null gracefully
            # Extract polygon coordinates
            regions = label.get('regions', [])  # Empty list if 'regions' is not present
            for region in regions:
                polygon = [(int(pt['x']), int(pt['y'])) for pt in region]
                polygons.append(polygon)
        return polygons
    def load_mask(self, image_id):
        # Override this method to load pixel-wise masks
        image_info = self.image_info[image_id]
        masks = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],dtype=np.uint8)
        class_ids = [self.class_names.index(cls) for cls in image_info["classes"]]
        class_ids = np.array(class_ids)
        for i, polygon in enumerate(image_info["polygons"]):
            # Extract polygon coordinates
            cc, rr = skimage.draw.polygon(np.array(polygon)[:, 0], np.array(polygon)[:, 1])
            # Ensure coordinates are within image bounds
            rr = np.clip(rr, 0, image_info["height"] - 1)
            cc = np.clip(cc, 0, image_info["width"] - 1)
            # Set pixels inside the polygon to 1
            masks[rr, cc, i] = 1
        masks = masks.astype(np.uint8)
        return masks, class_ids

# Example usage
dataset_dir = "C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/stills"
subsetsubfolder = ""
dataset = FishDataset()
dataset.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8)
dataset.prepare()



# Create a dataset instance for training
dataset_train = FishDataset()
dataset_train.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8)
dataset_train.prepare()

# Create a dataset instance for testing
dataset_test = FishDataset()
dataset_test.load_fish_dataset(dataset_dir, subsetsubfolder, split_ratio=0.8, is_training=False)
dataset_test.prepare()



# Access annotations for a specific image in the training set
train_image_id = 0
train_image_info = dataset_train.image_info[train_image_id]
print("Training Image ID: ", train_image_info["id"], "\nPolygons: ", train_image_info["polygons"])

# Access annotations for all images in the training set
for i in range(len(dataset_train.image_info)):
    train_image_info = dataset_train.image_info[i]
    print("Training Image ID: ", train_image_info["id"], "\nPolygons: ", train_image_info["polygons"])

# Access annotations for all images in the testing set
for i in range(len(dataset_train.image_info), len(dataset_train.image_info) + len(dataset_test.image_info)):
    test_image_info = dataset_test.image_info[i - len(dataset_train.image_info)]
    print("Testing Image ID: ", test_image_info["id"], "\nPolygons: ", test_image_info["polygons"])


# View a mask for a specific training image
train_image_id = 0
plt.imshow(dataset_train.load_mask(train_image_id)[0][:, :, 0], cmap='gray'), plt.show()



# View a mask for a specific testing image
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
                unique_classes.add(class_name.lower())  # Convert to lowercase for case-insensitive counting
    return unique_classes

nclasses = len(get_unique_classes(dataset_dir, subsetsubfolder))





class TrainingConfig(Config):
    """Configuration for training."""
    # Give the configuration a recognizable name
    NAME = "Training"
    # Train on 1 GPU and 2 images per GPU. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + nclasses  # background + n fish
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1280
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (40, 80, 160, 320, 640)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 25
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = TrainingConfig()
config.display()




# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
                          
                          
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

model_path = os.path.join(MODEL_DIR, "mask_rcnn_fish.h5")
model.keras_model.save_weights(model_path)



# Looking at performance on new/test data
class InferenceConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "mask_rcnn_fish.h5")

# Load trained weights
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_test.class_names, r['scores'], ax=get_ax())


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
