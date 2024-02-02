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
        self.add_class("fish", 1, "fish")
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
            self.add_image(
                "fish",
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
            if class_name and class_name.lower() == 'fish':  # Check for a valid class name
                # Extract polygon coordinates
                regions = label.get('regions', [])  # Empty list if 'regions' is not present
                for region in regions:
                    polygon = [(int(pt['x']), int(pt['y'])) for pt in region]
                    polygons.append(polygon)
        return polygons
    def load_mask(self, image_id):
        # Override this method to load pixel-wise masks
        image_info = self.image_info[image_id]
        # Skip if not a 'fish' source
        if image_info["source"] != "fish":
            return super(self.__class__, self).load_mask(image_id)
        masks = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                         dtype=np.uint8)
        class_ids = np.ones([len(image_info["polygons"])], dtype=np.int32)
        for i, polygon in enumerate(image_info["polygons"]):
            # Extract polygon coordinates
            rr, cc = skimage.draw.polygon(np.array(polygon)[:, 0], np.array(polygon)[:, 1])
            # Ensure coordinates are within image bounds
            rr = np.clip(rr, 0, image_info["height"] - 1)
            cc = np.clip(cc, 0, image_info["width"] - 1)
            # Set pixels inside the polygon to 1
            masks[rr, cc, i] = 1
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





def create_training_config(nclasses):
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
    return TrainingConfig()

config = create_training_config(nclasses)
config.display()




# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

