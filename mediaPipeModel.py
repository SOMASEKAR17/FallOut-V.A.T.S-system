import cv2
import mediapipe as mp
import numpy as np
import os
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
image = cv2.imread("./Input/man.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dot_spec = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=-1, circle_radius=5)
line_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3)
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(rgb)
h, w = image.shape[:2]
def get_pt(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])
def limb_length(landmarks, ids):
    """Total pixel length along a chain of joints."""
    pts = [get_pt(landmarks, i) for i in ids]
    return sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
def estimate_radii(landmarks):
    """
    Derive pipe radii from actual limb lengths and eye distance.
    Strategy:
      - Eye distance  -> distance proxy (how far person is)
      - Limb length   -> how long the bone is on screen
      - radius = limb_length * ratio  (ratio tuned per body part)
    The ratio expresses 'what fraction of the limb length is the limb width'.
    Anatomically:
      upper arm  ~20% of its length
      forearm    ~18%
      thigh      ~25%
      shin       ~20%
      torso      ~40% of torso height
    We blend eye-distance scale in as a sanity check.
    """
    l_eye = get_pt(landmarks, 2)
    r_eye = get_pt(landmarks, 5)
    eye_dist = np.linalg.norm(l_eye - r_eye)
    eye_scale = eye_dist / 100.0   
    r_upper_arm = limb_length(landmarks, [12, 14])
    r_forearm   = limb_length(landmarks, [14, 16])
    l_upper_arm = limb_length(landmarks, [11, 13])
    l_forearm   = limb_length(landmarks, [13, 15])
    r_thigh = limb_length(landmarks, [24, 26])
    r_shin  = limb_length(landmarks, [26, 28])
    l_thigh = limb_length(landmarks, [23, 25])
    l_shin  = limb_length(landmarks, [25, 27])
    torso_h = limb_length(landmarks, [11, 23])   
    torso_w = limb_length(landmarks, [11, 12])   
    def r(length, ratio):
        raw = length * ratio
        blended = (raw + (ratio * 100 * eye_scale)) / 2
        return max(int(blended), 10)
    return {
        "r_upper_arm" : r(r_upper_arm, 0.22),
        "r_forearm"   : r(r_forearm,   0.18),
        "l_upper_arm" : r(l_upper_arm, 0.22),
        "l_forearm"   : r(l_forearm,   0.18),
        "r_thigh"     : r(r_thigh,     0.28),
        "r_shin"      : r(r_shin,      0.22),
        "l_thigh"     : r(l_thigh,     0.28),
        "l_shin"      : r(l_shin,      0.22),
        "body_pad"    : r(torso_w,     0.30),
    }
def draw_pipe_segment(img, p1, p2, radius, color, alpha=0.25):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1:
        return
    perp = np.array([-direction[1], direction[0]]) / length
    angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))
    c1 = (p1 + perp * radius).astype(int)
    c2 = (p1 - perp * radius).astype(int)
    c3 = (p2 - perp * radius).astype(int)
    c4 = (p2 + perp * radius).astype(int)
    box = np.array([c1, c2, c3, c4])
    overlay = img.copy()
    cv2.fillPoly(overlay, [box], color)
    cv2.ellipse(overlay, tuple(p1.astype(int)), (radius, radius), angle_deg,  90, 270, color, -1)
    cv2.ellipse(overlay, tuple(p2.astype(int)), (radius, radius), angle_deg, -90,  90, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [box], isClosed=False, color=color, thickness=2)
    cv2.ellipse(img, tuple(p1.astype(int)), (radius, radius), angle_deg,  90, 270, color, 2)
    cv2.ellipse(img, tuple(p2.astype(int)), (radius, radius), angle_deg, -90,  90, color, 2)
def draw_face_circle(img, landmarks, color, alpha=0.3):
    nose  = get_pt(landmarks, 0)
    l_eye = get_pt(landmarks, 2)
    r_eye = get_pt(landmarks, 5)
    eye_dist = np.linalg.norm(l_eye - r_eye)
    radius = int(eye_dist * 2.5)
    center = nose.astype(int)
    overlay = img.copy()
    cv2.circle(overlay, tuple(center), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, tuple(center), radius, color, 2)
def draw_body_outline(img, landmarks, color, padding=35, alpha=0.25):
    ids = [11, 12, 24, 23]
    pts = np.array([get_pt(landmarks, i) for i in ids], dtype=np.float64)
    centroid = pts.mean(axis=0)
    expanded = []
    for pt in pts:
        d = pt - centroid
        n = np.linalg.norm(d)
        expanded.append((pt + d / n * padding).astype(int) if n > 0 else pt.astype(int))
    expanded = np.array(expanded, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [expanded], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [expanded], isClosed=True, color=color, thickness=2)
    corner_r = max(int(padding * 0.3), 8)
    for pt in expanded:
        o2 = img.copy()
        cv2.circle(o2, tuple(pt), corner_r, color, -1)
        cv2.addWeighted(o2, alpha, img, 1 - alpha, 0, img)
        cv2.circle(img, tuple(pt), corner_r, color, 2)
if results.pose_landmarks:
    lm = results.pose_landmarks.landmark
    radii = estimate_radii(lm)
    print("Computed radii:", radii)
    draw_face_circle(image, lm, color=(255, 200, 0), alpha=0.25)
    draw_body_outline(image, lm, color=(200, 0, 200),
                      padding=radii["body_pad"], alpha=0.25)
    r_shoulder = get_pt(lm, 12)
    r_elbow    = get_pt(lm, 14)
    r_wrist    = get_pt(lm, 16)
    draw_pipe_segment(image, r_shoulder, r_elbow, radii["r_upper_arm"], (0, 128, 255))
    draw_pipe_segment(image, r_elbow,    r_wrist, radii["r_forearm"],   (0, 100, 220))
    l_shoulder = get_pt(lm, 11)
    l_elbow    = get_pt(lm, 13)
    l_wrist    = get_pt(lm, 15)
    draw_pipe_segment(image, l_shoulder, l_elbow, radii["l_upper_arm"], (0, 200, 100))
    draw_pipe_segment(image, l_elbow,    l_wrist, radii["l_forearm"],   (0, 160,  80))
    r_hip   = get_pt(lm, 24)
    r_knee  = get_pt(lm, 26)
    r_ankle = get_pt(lm, 28)
    r_foot  = get_pt(lm, 32)
    draw_pipe_segment(image, r_hip,   r_knee,  radii["r_thigh"], (0,  50, 255))
    draw_pipe_segment(image, r_knee,  r_ankle, radii["r_shin"],  (0,  80, 200))
    draw_pipe_segment(image, r_ankle, r_foot,  radii["r_shin"],  (0, 100, 180))
    l_hip   = get_pt(lm, 23)
    l_knee  = get_pt(lm, 25)
    l_ankle = get_pt(lm, 27)
    l_foot  = get_pt(lm, 31)
    draw_pipe_segment(image, l_hip,   l_knee,  radii["l_thigh"], (255, 100,   0))
    draw_pipe_segment(image, l_knee,  l_ankle, radii["l_shin"],  (220,  80,   0))
    draw_pipe_segment(image, l_ankle, l_foot,  radii["l_shin"],  (180,  60,   0))
    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=dot_spec,
        connection_drawing_spec=line_spec
    )
os.makedirs("./Output", exist_ok=True)
cv2.imwrite("./Output/man_pose.jpg", image)
scale_disp = min(900 / h, 900 / w, 1.0)
display = cv2.resize(image, (int(w * scale_disp), int(h * scale_disp)))
cv2.namedWindow("Output-Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output-Pose", display.shape[1], display.shape[0])
cv2.imshow("Output-Pose", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
