# Fallout V.A.T.S. Themed Part Detection

This project is inspired by the **V.A.T.S.** (Vault-Tec Assisted Targeting System) from the *Fallout* game series. It utilizes **MediaPipe** and **OpenCV** to detect human pose and segment specific body parts, highlighting them in a fluorescent glowing green color reminiscent of the in-game targeting interface.

##  Features

- **Webcam Image Capture**: Interactively snap an image directly from your webcam.
- **Pose Estimation & Segmentation**: Powered by MediaPipe's robust models to map out 33 skeletal landmarks along with a dense segmentation mask.
- **V.A.T.S Visual Style**: Procedurally builds dynamic masks for distinct body parts (Head, Torso, Left/Right Arms, Left/Right Legs) and highlights them with the classic "Fallout Green."
- **Interactive Part Selection**: Navigate through different segmented body parts visually, mirroring how players cycle through targets in V.A.T.S.

##  Model Implementations

This repository includes several distinct scripts that explore different techniques for human posture and part detection:

- **`main.py`** 
  The primary interactive pipeline. It uses a combination of MediaPipe Pose and MediaPipe Vision Segmenter to precisely mask out and draw VATS-style outlines around the user.
- **`mediaPipeModel.py`**
  A MediaPipe-based openpose implementation focused strictly on human posture mapping, constructing spatial bounds for body parts.
- **`mediaPipeSegmentationModel.py`**
  An implementation dedicated to the segmentation model for extracting and rendering green-highlighted body parts smoothly on top of a camera feed.
- **`mpiCaffeModel.py`**
  An alternative implementation utilizing an **MPI Caffe Model** for pose detection. It predicts 15 keypoints (compared to MediaPipe's 33). *Note: See requirements below if you choose to run this script.*

##  Installation & Usage

1. **Create and Activate a Virtual Environment:**
   *(Windows - PowerShell)*
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(macOS / Linux)*
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   Install required packages via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Execute the primary interactive script:
   ```bash
   python main.py
   ```
   *Controls in `main.py`:*
   - `<SPACE>`: Capture image via webcam.
   - `D`: Next body part.
   - `A`: Previous body part.
   - `Q` or `ESC`: Quit / Close window.

##  Additional Models

If you wish to experiment with `mpiCaffeModel.py` instead of the MediaPipe implementations, you must manually download the corresponding `.caffemodel` weights, as they are too large for this repository.
Download `pose_iter_160000.caffemodel` from Hugging Face and place it within the root of this project directory:
**[Download MPI Caffe Model Here](https://huggingface.co/camenduru/openpose/blob/f4a22b0e6fa2a4a2b1e2d50bd589e8bb11ebea7c/pose_iter_160000.caffemodel)**
