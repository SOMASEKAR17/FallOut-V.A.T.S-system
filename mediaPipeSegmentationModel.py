import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageSegmenterOptions, RunningMode
import urllib.request
import os
MODEL_PATH = "selfie_multiclass_256x256.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
INPUT_IMAGE = "./Input/man.jpg"
OUTPUT_DIR  = "./Output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "night_vision.png")  
NV_GREEN = (0, 255, 70)  
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
def create_segmenter():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = ImageSegmenterOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)
def segment_image(segmenter, image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)
    category_mask = (
        result.category_mask.numpy_view().copy()
        if result.category_mask is not None else None
    )
    confidence_masks = (
        [m.numpy_view().copy() for m in result.confidence_masks]
        if result.confidence_masks else []
    )
    return category_mask, confidence_masks
def apply_night_vision_transparent(image_bgr, category_mask, confidence_masks):
    h, w = image_bgr.shape[:2]
    if confidence_masks and len(confidence_masks) > 1:
        soft_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(1, len(confidence_masks)):   
            soft_mask += confidence_masks[i]
        soft_mask = np.clip(soft_mask, 0.0, 1.0)
    else:
        soft_mask = (category_mask != 0).astype(np.float32)
    soft_mask = cv2.GaussianBlur(soft_mask, (5, 5), 0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out = np.zeros((h, w, 4), dtype=np.uint8)
    green_intensity = np.clip(gray * 1.4 + 0.15, 0.0, 1.0)
    out[:, :, 0] = (NV_GREEN[0] * green_intensity).astype(np.uint8)   
    out[:, :, 1] = (NV_GREEN[1] * green_intensity).astype(np.uint8)   
    out[:, :, 2] = (NV_GREEN[2] * green_intensity).astype(np.uint8)   
    out[:, :, 3] = (soft_mask * 255).astype(np.uint8)                 
    return out
def process():
    download_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_bgr = cv2.imread(INPUT_IMAGE)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read: {INPUT_IMAGE}")
    print(f"Loaded: {INPUT_IMAGE}  ({image_bgr.shape[1]}×{image_bgr.shape[0]})")
    with create_segmenter() as segmenter:
        category_mask, confidence_masks = segment_image(segmenter, image_bgr)
    print(f"Classes detected: {np.unique(category_mask)}")
    result = apply_night_vision_transparent(image_bgr, category_mask, confidence_masks)
    cv2.imwrite(OUTPUT_FILE, result)   
    print(f"Saved → {OUTPUT_FILE}")
if __name__ == "__main__":
    process()
