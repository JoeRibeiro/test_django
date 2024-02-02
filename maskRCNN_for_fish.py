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


# from mrcnn.config import Config
# class CocoConfig(Config):
    # """Configuration for training on MS COCO.
    # Derives from the base Config class and overrides values specific
    # to the COCO dataset.
    # """
    # # Give the configuration a recognizable name
    # NAME = "coco"
    # # We use a GPU with 12GB memory, which can fit two images.
    # # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 2
    # # Uncomment to train on 8 GPUs (default is 1)
    # # GPU_COUNT = 8
    # # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes



class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()




import os
import json
import numpy as np
import skimage.draw
from PIL import Image  # Pillow library
from mrcnn import utils

class FishDataset(utils.Dataset):
    def load_fish_dataset(self, dataset_dir, subset):
        # Add classes (adjust based on your dataset)
        self.add_class("fish", 1, "fish")
        # Define data directory
        data_dir = os.path.join(dataset_dir, subset)
        # List all files in the data directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        for json_file in files:
            # Load JSON file
            with open(os.path.join(data_dir, json_file)) as f:
                data = json.load(f)
            # Extract image details
            image_path = os.path.join(data_dir, data['image_filename'])
            image_id = os.path.splitext(json_file)[0]  # Extract image ID from filename
            # Get image width and height
            image = Image.open(image_path)
            width, height = image.size
            # Add image to dataset
            self.add_image(
                "fish",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                polygons=self.parse_polygons(data['labels'])
            )
    def parse_polygons(self, labels):
        polygons = []
        for label in labels:
            class_name = label['label_class']
            if class_name == 'fish':  # Adjust based on your class labels
                print("Processing label: ", label)
                # Extract polygon coordinates
                regions = label.get('regions', [])  # Empty list if 'regions' is not present
                for region in regions:
                    print("Processing region: ", region)
                    polygon = [(int(pt['x']), int(pt['y'])) for pt in region]
                    polygons.append(polygon)
        return polygons
    def load_mask(self, image_id):
        # Override this method to load mask annotations
        image_info = self.image_info[image_id]
        if image_info["source"] != "fish":
            return super(self.__class__, self).load_mask(image_id)
        masks = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                         dtype=np.uint8)
        class_ids = np.ones([len(image_info["polygons"])], dtype=np.int32)
        for i, polygon in enumerate(image_info["polygons"]):
            rr, cc = skimage.draw.polygon(np.array(polygon)[:, 1], np.array(polygon)[:, 0])
            masks[rr, cc, i] = 1
        return masks, class_ids
    def load_fish(self, image_id):
        # Load the fish annotations for a specific image
        image_info = self.image_info[image_id]
        fish_polygons = image_info.get("polygons", [])
        return fish_polygons

# Example usage
dataset = FishDataset()
dataset.load_fish_dataset("C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/stills", "")
dataset.prepare()

# Access annotations for a specific image
image_id = 0
image_info = dataset.image_info[image_id]
print("Image ID: ", image_info["id"], "\nPolygons: ", image_info["polygons"])


