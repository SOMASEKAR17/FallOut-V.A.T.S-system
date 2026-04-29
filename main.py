import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageSegmenterOptions, RunningMode
MODEL_PATH  = os.path.join(BASE_DIR, "selfie_multiclass_256x256.tflite")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
OUTPUT_DIR  = os.path.join(BASE_DIR, "Output")
NV_GREEN    = (0, 255, 70)
def capture_from_webcam():
    """Open webcam, show live feed, capture on SPACE, quit on ESC/Q."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam open. Press SPACE to capture, ESC/Q to quit.")
    cv2.namedWindow("Webcam — press SPACE to capture", cv2.WINDOW_NORMAL)
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break
        display = cv2.flip(frame, 1)
        cv2.putText(display, "SPACE = capture   ESC/Q = quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, "SPACE = capture   ESC/Q = quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.imshow("Webcam — press SPACE to capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured = cv2.flip(frame, 1)
            print("Image captured.")
            break
        elif key in (27, ord('q')):
            print("Cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading segmentation model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
def create_segmenter():
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = ImageSegmenterOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)
def segment_image(segmenter, image_bgr):
    rgb      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = segmenter.segment(mp_image)
    category_mask = (
        result.category_mask.numpy_view().copy()
        if result.category_mask is not None else None
    )
    confidence_masks = (
        [m.numpy_view().copy() for m in result.confidence_masks]
        if result.confidence_masks else []
    )
    return category_mask, confidence_masks
def build_seg_soft_mask(image_bgr, category_mask, confidence_masks):
    h, w = image_bgr.shape[:2]
    if confidence_masks and len(confidence_masks) > 1:
        seg_soft = np.zeros((h, w), dtype=np.float32)
        for i in range(1, len(confidence_masks)):
            seg_soft += confidence_masks[i]
        seg_soft = np.clip(seg_soft, 0.0, 1.0)
    else:
        seg_soft = (category_mask != 0).astype(np.float32)
    if seg_soft.shape != (h, w):
        seg_soft = cv2.resize(seg_soft, (w, h), interpolation=cv2.INTER_LINEAR)
    return seg_soft
def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])
def limb_length(landmarks, ids, w, h):
    pts = [get_pt(landmarks, i, w, h) for i in ids]
    return sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
def estimate_radii(landmarks, w, h):
    l_eye     = get_pt(landmarks, 2, w, h)
    r_eye     = get_pt(landmarks, 5, w, h)
    eye_dist  = np.linalg.norm(l_eye - r_eye)
    eye_scale = eye_dist / 100.0
    def r(length, ratio):
        raw     = length * ratio
        blended = (raw + ratio * 100 * eye_scale) / 2
        return max(int(blended), 10)
    return {
        "r_upper_arm" : r(limb_length(landmarks, [12, 14], w, h), 0.45), 
        "r_forearm"   : r(limb_length(landmarks, [14, 16], w, h), 0.40), 
        "l_upper_arm" : r(limb_length(landmarks, [11, 13], w, h), 0.45), 
        "l_forearm"   : r(limb_length(landmarks, [13, 15], w, h), 0.40),  
        "r_thigh"     : r(limb_length(landmarks, [24, 26], w, h), 0.55), 
        "r_shin"      : r(limb_length(landmarks, [26, 28], w, h), 0.45),  
        "l_thigh"     : r(limb_length(landmarks, [23, 25], w, h), 0.55),  
        "l_shin"      : r(limb_length(landmarks, [25, 27], w, h), 0.45), 
        "body_pad"    : r(limb_length(landmarks, [11, 12], w, h), 0.65),  
        "face_radius" : int(eye_dist * 2.5),                               
        "face_center" : get_pt(landmarks, 0, w, h).astype(int),
    }
def make_face_mask(radii, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, tuple(radii["face_center"]), radii["face_radius"], 255, -1)
    return mask
def make_pipe_mask(p1, p2, radius, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    p1   = np.array(p1, dtype=np.float64)
    p2   = np.array(p2, dtype=np.float64)
    direction = p2 - p1
    length    = np.linalg.norm(direction)
    if length < 1:
        return mask
    perp      = np.array([-direction[1], direction[0]]) / length
    angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))
    c1 = (p1 + perp * radius).astype(int)
    c2 = (p1 - perp * radius).astype(int)
    c3 = (p2 - perp * radius).astype(int)
    c4 = (p2 + perp * radius).astype(int)
    cv2.fillPoly(mask, [np.array([c1, c2, c3, c4])], 255)
    cv2.ellipse(mask, tuple(p1.astype(int)), (radius, radius), angle_deg,  90, 270, 255, -1)
    cv2.ellipse(mask, tuple(p2.astype(int)), (radius, radius), angle_deg, -90,  90, 255, -1)
    return mask
def make_body_mask(landmarks, padding, shape, w, h):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    ids  = [11, 12, 24, 23]
    pts  = np.array([get_pt(landmarks, i, w, h) for i in ids], dtype=np.float64)
    centroid = pts.mean(axis=0)
    expanded = []
    for pt in pts:
        d = pt - centroid
        n = np.linalg.norm(d)
        expanded.append((pt + d / n * padding).astype(int) if n > 0 else pt.astype(int))
    cv2.fillPoly(mask, [np.array(expanded, dtype=np.int32)], 255)
    return mask
def combine_masks(mask_list, shape):
    combined = np.zeros(shape[:2], dtype=np.uint8)
    for m in mask_list:
        combined = cv2.bitwise_or(combined, m)
    return combined


def draw_pipe_outline(img, p1, p2, radius, color, alpha=0.18):
    pass  

def draw_face_outline(img, radii, color, alpha=0.18):
    pass 

def draw_body_outline(img, landmarks, padding, color, w, h, alpha=0.18):
    pass  

def get_mask_bbox(mask):
    """Get bounding box of a binary mask."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)
    return x, y, bw, bh

def draw_crosshair(img, mask, label, color):
    """
    Draw a rectangle outline + center dot + label
    around the bounding box of the mask.
    Color matches the active part.
    """
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return

    x, y, bw, bh = bbox
    pad = 20
    x1, y1 = x - pad, y - pad
    x2, y2 = x + bw + pad, y + bh + pad

   
    ih, iw = img.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, iw - 1), min(y2, ih - 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    thickness = 2
    corner_len = 30  
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.circle(img, (cx, cy), 6, color, -1)
    cv2.line(img, (cx - 15, cy), (cx + 15, cy), color, thickness)
    cv2.line(img, (cx, cy - 15), (cx, cy + 15), color, thickness)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
    lx = cx - tw // 2
    ly = y1 - 10
    ly = max(ly, th + 5)  

    cv2.putText(img, label, (lx, ly), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), font, font_scale, color,     2, cv2.LINE_AA)

def apply_green_highlight(base_img, part_mask, seg_soft, highlight_color=NV_GREEN):
    out       = base_img.copy()
    part_norm = part_mask.astype(np.float32) / 255.0
    combined  = part_norm * seg_soft
    combined  = cv2.GaussianBlur(combined, (7, 7), 0)
    gray      = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    green_int = np.clip(gray * 1.4 + 0.15, 0.0, 1.0)
    for c, nv_val in enumerate(highlight_color):
        ch = out[:, :, c].astype(np.float32)
        gc = nv_val * green_int
        out[:, :, c] = np.clip(ch * (1 - combined) + gc * combined, 0, 255).astype(np.uint8)
    return out

def draw_label(img, text, pos, color, scale=1.0):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   2, cv2.LINE_AA)

def run_pipeline(image_bgr):
    h, w = image_bgr.shape[:2]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processing = image_bgr.copy()
    cv2.putText(processing, "Processing, please wait...",
                (w // 2 - 300, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(processing, "Processing, please wait...",
                (w // 2 - 300, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 120), 3, cv2.LINE_AA)
    sc   = min(900 / h, 900 / w, 1.0)
    disp = cv2.resize(processing, (int(w*sc), int(h*sc)))
    cv2.namedWindow("Part Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Part Detection", disp.shape[1], disp.shape[0])
    cv2.imshow("Part Detection", disp)
    cv2.waitKey(1)
    print("Running pose estimation...")
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        pose_results = pose.process(rgb)
    if not pose_results.pose_landmarks:
        raise RuntimeError("No pose detected in captured image. Try again.")
    lm    = pose_results.pose_landmarks.landmark
    radii = estimate_radii(lm, w, h)
    print("Radii:", radii)
    print("Running segmentation...")
    with create_segmenter() as segmenter:
        category_mask, confidence_masks = segment_image(segmenter, image_bgr)
    seg_soft = build_seg_soft_mask(image_bgr, category_mask, confidence_masks)
    print("Segmentation done.")
    r_shoulder = get_pt(lm, 12, w, h);  r_elbow = get_pt(lm, 14, w, h)
    r_wrist    = get_pt(lm, 16, w, h)
    l_shoulder = get_pt(lm, 11, w, h);  l_elbow = get_pt(lm, 13, w, h)
    l_wrist    = get_pt(lm, 15, w, h)
    r_hip      = get_pt(lm, 24, w, h);  r_knee  = get_pt(lm, 26, w, h)
    r_ankle    = get_pt(lm, 28, w, h);  r_foot  = get_pt(lm, 32, w, h)
    l_hip      = get_pt(lm, 23, w, h);  l_knee  = get_pt(lm, 25, w, h)
    l_ankle    = get_pt(lm, 27, w, h);  l_foot  = get_pt(lm, 31, w, h)
    shape      = image_bgr.shape
    parts = [
        {
            "name"  : "Head",
            "color" : (0, 220, 255),
            "mask"  : make_face_mask(radii, shape),
            "draw"  : lambda img, c=(0, 220, 255):
                          draw_face_outline(img, radii, c),
        },
        {
            "name"  : "Right Arm",
            "color" : (0, 128, 255),
            "mask"  : combine_masks([
                make_pipe_mask(r_shoulder, r_elbow, radii["r_upper_arm"], shape),
                make_pipe_mask(r_elbow,    r_wrist, radii["r_forearm"],   shape),
            ], shape),
            "draw"  : lambda img, c=(0, 128, 255): [
                draw_pipe_outline(img, r_shoulder, r_elbow, radii["r_upper_arm"], c),
                draw_pipe_outline(img, r_elbow,    r_wrist, radii["r_forearm"],   c),
            ],
        },
        {
            "name"  : "Left Arm",
            "color" : (0, 200, 100),
            "mask"  : combine_masks([
                make_pipe_mask(l_shoulder, l_elbow, radii["l_upper_arm"], shape),
                make_pipe_mask(l_elbow,    l_wrist, radii["l_forearm"],   shape),
            ], shape),
            "draw"  : lambda img, c=(0, 200, 100): [
                draw_pipe_outline(img, l_shoulder, l_elbow, radii["l_upper_arm"], c),
                draw_pipe_outline(img, l_elbow,    l_wrist, radii["l_forearm"],   c),
            ],
        },
        {
            "name"  : "Body",
            "color" : (200, 0, 200),
            "mask"  : make_body_mask(lm, radii["body_pad"], shape, w, h),
            "draw"  : lambda img, c=(200, 0, 200):
                          draw_body_outline(img, lm, radii["body_pad"], c, w, h),
        },
        {
            "name"  : "Right Leg",
            "color" : (0, 50, 255),
            "mask"  : combine_masks([
                make_pipe_mask(r_hip,   r_knee,  radii["r_thigh"], shape),
                make_pipe_mask(r_knee,  r_ankle, radii["r_shin"],  shape),
                make_pipe_mask(r_ankle, r_foot,  radii["r_shin"],  shape),
            ], shape),
            "draw"  : lambda img, c=(0, 50, 255): [
                draw_pipe_outline(img, r_hip,   r_knee,  radii["r_thigh"], c),
                draw_pipe_outline(img, r_knee,  r_ankle, radii["r_shin"],  c),
                draw_pipe_outline(img, r_ankle, r_foot,  radii["r_shin"],  c),
            ],
        },
        {
            "name"  : "Left Leg",
            "color" : (255, 100, 0),
            "mask"  : combine_masks([
                make_pipe_mask(l_hip,   l_knee,  radii["l_thigh"], shape),
                make_pipe_mask(l_knee,  l_ankle, radii["l_shin"],  shape),
                make_pipe_mask(l_ankle, l_foot,  radii["l_shin"],  shape),
            ], shape),
            "draw"  : lambda img, c=(255, 100, 0): [
                draw_pipe_outline(img, l_hip,   l_knee,  radii["l_thigh"], c),
                draw_pipe_outline(img, l_knee,  l_ankle, radii["l_shin"],  c),
                draw_pipe_outline(img, l_ankle, l_foot,  radii["l_shin"],  c),
            ],
        },
    ]
    def build_frame(part_idx):
        part  = parts[part_idx]
        frame = apply_green_highlight(image_bgr.copy(), part["mask"], seg_soft)
        draw_crosshair(frame, part["mask"], part["name"], NV_GREEN)
        draw_label(frame, "A=Prev  D=Next  Q=Quit", (30, h - 30), (180, 180, 180), scale=0.8)
        return frame


    DISPLAY_MAX = 900
    current     = 0
    def show(frame):
        sc   = min(DISPLAY_MAX / frame.shape[0], DISPLAY_MAX / frame.shape[1], 1.0)
        disp = cv2.resize(frame, (int(frame.shape[1]*sc), int(frame.shape[0]*sc)))
        cv2.resizeWindow("Part Detection", disp.shape[1], disp.shape[0])
        cv2.imshow("Part Detection", disp)
    show(build_frame(current))
    print("Controls: A = previous | D = next | Q / ESC = quit")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            current = (current + 1) % len(parts)
        elif key == ord('a'):
            current = (current - 1) % len(parts)
        else:
            continue
        frame    = build_frame(current)
        out_path = os.path.join(OUTPUT_DIR,
                    f"part_{parts[current]['name'].replace(' ', '_')}.jpg")
        cv2.imwrite(out_path, frame)
        show(frame)
    cv2.destroyAllWindows()
def main():
    download_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_bgr = capture_from_webcam()
    if image_bgr is None:
        print("No image captured. Exiting.")
        return
    raw_path = os.path.join(OUTPUT_DIR, "captured.jpg")
    cv2.imwrite(raw_path, image_bgr)
    print(f"Capture saved to {raw_path}")
    run_pipeline(image_bgr)
if __name__ == "__main__":
    main()
