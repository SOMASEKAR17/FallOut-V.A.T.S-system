import cv2
import numpy as np
import os
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
image = cv2.imread("./Input/man.jpg")
if image is None:
    raise FileNotFoundError("Image not found. Check the filename and path.")
frameWidth = image.shape[1]
frameHeight = image.shape[0]
lineStrictness=270
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (lineStrictness,lineStrictness),
                                (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()
BODY_PARTS = {
    0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest"
}
POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7],
    [1, 14], [14, 8], [8, 9], [9, 10],
    [14, 11], [11, 12], [12, 13]
]
points = []
for i in range(15):
    probMap = output[0, i, :, :]
    _, prob, _, point = cv2.minMaxLoc(probMap)
    x = (frameWidth * point[0]) / output.shape[3]
    y = (frameHeight * point[1]) / output.shape[2]
    if prob > 0.1:
        points.append((int(x), int(y)))
    else:
        points.append(None)
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    if points[partA] and points[partB]:
        cv2.line(image, points[partA], points[partB], (0, 255, 0), 15)
for i, point in enumerate(points):
    if point:
        cv2.circle(image, point, 6, (0, 255, 255), thickness=5, lineType=cv2.FILLED)
        cv2.putText(image, BODY_PARTS[i], point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
os.makedirs("./Output", exist_ok=True)
output_path = "./Output/man_pose.jpg"
cv2.imwrite(output_path, image)
print(f"Saved to {output_path}")
DISPLAY_MAX = 900  
scale = min(DISPLAY_MAX / frameHeight, DISPLAY_MAX / frameWidth, 1.0)
display = cv2.resize(image, (int(frameWidth * scale), int(frameHeight * scale)))
cv2.namedWindow("Output-Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output-Pose", display.shape[1], display.shape[0])
cv2.imshow("Output-Pose", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
