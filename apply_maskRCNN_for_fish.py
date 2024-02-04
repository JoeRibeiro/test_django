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

ROOT_DIR = os.path.abspath("C:/Users/JR13/OneDrive - CEFAS/My onedrive documents/test_django/")
SUBFOLDER = ""
STILLS_DIR = os.path.join(ROOT_DIR, "stills")
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
MASKRCNN_DIR = os.path.join(ROOT_DIR, "maskrcnn")
LOGS_DIR = os.path.join(MASKRCNN_DIR, "logs")
COCO_MODEL_PATH = os.path.join(MASKRCNN_DIR, "mask_rcnn_coco.h5")
VIDEO_OUT_DIR = os.path.join(ROOT_DIR, "processed_videos")
MODEL_DIR = os.path.join(MASKRCNN_DIR, "logs")
dataset_dir = STILLS_DIR
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

config = TrainingConfig()


class InferenceConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


# Create a new instance of the Mask R-CNN model
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Load the weights from the saved model file
model.load_weights(os.path.join(MODEL_DIR, "mask_rcnn_fish.h5"), by_name=True)



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
    # Generate consistent random colors for each class
    class_colors = {class_id: tuple(np.random.randint(0, 256, 3, dtype=np.uint8)) for class_id in range(config.NUM_CLASSES)}
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.detect([frame], verbose=0)
        r = results[0]
        # Draw bounding boxes and masks with class-specific colors
        for i in range(r['rois'].shape[0]):
            class_id = r['class_ids'][i]
            color = class_colors[class_id]
            color = tuple(map(int, color))
            box = r['rois'][i]
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), color, 2)
            mask = r['masks'][:, :, i]
            frame[mask > 0] = frame[mask > 0] * 0.5 * 0.5
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.MP4')]
for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    output_path = os.path.join(VIDEO_OUT_DIR, f"output_{video_file}")
    process_video(video_path, output_path, num_frames = 500)
