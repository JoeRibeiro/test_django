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
dataset_dir = "C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/stills"
subsetsubfolder = ""

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

class InferenceConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


# Create a new instance of the Mask R-CNN model
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Load the weights from the saved model file
model.load_weights(os.path.join(MODEL_DIR, "mask_rcnn_fish.h5"), by_name=True)

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


def process_video(video_path, output_path, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is not None:
        frame_count = min(frame_count, num_frames)
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.detect([frame], verbose=0)
        r = results[0]
        # Draw bounding boxes
        for i in range(r['rois'].shape[0]):
            color = (0, 255, 0)  # Green color for bounding boxes
            box = r['rois'][i]
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), color, 2)
        # Draw masks
        for i in range(r['masks'].shape[2]):
            mask = r['masks'][:, :, i]
            color = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            frame[mask > 0] = frame[mask > 0] * 0.5 + color * 0.5
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_files = [f for f in os.listdir(VIDEO_IN_DIR) if f.endswith('.MP4')]
for video_file in video_files:
    video_path = os.path.join(VIDEO_IN_DIR, video_file)
    output_path = os.path.join(VIDEO_OUT_DIR, f"output_{video_file}")
    process_video(video_path, output_path, num_frames = 10)
