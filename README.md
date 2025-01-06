# Person Tracker with Face Recognition

This repository contains a Python-based implementation for real-time person tracking and face recognition using a combination of YOLOv11 for person detection and InsightFace for face recognition. The system uses threading for efficient logging and frame-saving, and it tracks known and unknown individuals in a video stream.

---

### Here's the general flow of the system:

![alt text](https://github.com/abdul-raouf/Person-Tracker-with-Face-Recognition/blob/main/flow_diagram.svg)

## **Table of Contents**
1. [Installation and Setup](#installation-and-setup)
2. [System Configuration](#system-configuration)
3. [Core Components](#core-components)
   - [YOLO Object Detection](#yolo-object-detection)
   - [Face Recognition](#face-recognition)
   - [Track Management](#track-management)
   - [Logging and Frame Saving](#logging-and-frame-saving)
   - [Generating Face Embeddings](#generating-face-embeddings)
4. [How It Works](#how-it-works)
5. [Running the Script](#running-the-script)

---

## **Installation and Setup**

1. **Install Required Libraries:**

```bash
pip install opencv-python opencv-contrib-python numpy torch transformers scipy insightface ultralytics
```

2. **Install Additional Tools:**
   - Install InsightFace:
     ```bash
     pip install insightface
     ```
   - Install YOLOv8:
     ```bash
     pip install ultralytics
     ```

3. **Prepare Your Environment:**
   - Ensure CUDA is installed for GPU acceleration.
   - Place the YOLOv8 model file and face embeddings in the correct paths.

---

## **System Configuration**

The script is pre-configured with several paths and parameters:
- **YOLO Model Path:** Path to the YOLOv8 weights file.
- **Known Embeddings Path:** Precomputed face embeddings.
- **Stream URL:** Video stream or file path for processing.

Modify the following configuration variables as needed:

```python
YOLO_MODEL_PATH = "path_to_yolo_model.pt"
KNOWN_EMBEDDINGS_PATH = "path_to_embeddings.npy"
KNOWN_LABELS_PATH = "path_to_labels.npy"
STREAM_URL = "path_to_video_stream"
DATA_SAVE_DIR = "path_to_save_directory"
```

---

## **Core Components**

### **1. YOLO Object Detection**

The system uses YOLOv8 for detecting human bounding boxes in the video frame. YOLO is initialized with:

```python
yolo_model = YOLO(YOLO_MODEL_PATH)
```

For each frame, detections are obtained with:

```python
yolo_results = yolo_model(frame, classes=[0], conf=0.4, iou=0.5)
```

### **2. Face Recognition**

The face recognition system leverages InsightFace to identify known individuals. The model is initialized and prepared as follows:

```python
face_recognition_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_recognition_model.prepare(ctx_id=0, det_size=(640, 640))
```

Each face embedding is matched against known embeddings using cosine similarity:

```python
similarities = [1 - cosine(embedding, known_emb) for known_emb in known_embeddings]
```

### **3. Track Management**

The system uses a custom `TrackManager` class to maintain tracks for detected individuals. Tracks are updated based on IOU (Intersection Over Union) matching:

```python
cost_matrix = [[1 - self._calculate_iou(track.bbox, det) for det in detections] for track in self.tracks]
```

Unmatched detections create new tracks, and tracks with no recent updates are removed.

### **4. Logging and Frame Saving**

Thread-safe queues manage logging and frame saving efficiently. Logging entries and frames are processed in separate threads:

```python
log_thread = Thread(target=log_writer)
frame_thread = Thread(target=frame_saver)
log_thread.start()
frame_thread.start()
```

### **5. Generating Face Embeddings**

To recognize faces, you need to generate face embeddings for known individuals. Follow these steps:

1. **Prepare Face Images:**
   - Collect images of known individuals and organize them into folders, one folder per individual.

2. **Load the InsightFace Model:**
   ```python
   from insightface.app import FaceAnalysis
   import numpy as np

   face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
   face_model.prepare(ctx_id=0, det_size=(640, 640))
   ```

3. **Extract Face Embeddings:**
   ```python
   import os
   from PIL import Image

   known_embeddings = []
   known_labels = []

   base_path = "path_to_known_faces"
   for person_name in os.listdir(base_path):
       person_folder = os.path.join(base_path, person_name)
       for img_file in os.listdir(person_folder):
           img_path = os.path.join(person_folder, img_file)
           img = Image.open(img_path).convert('RGB')
           faces = face_model.get(np.array(img))

           if faces:
               embedding = faces[0].embedding
               known_embeddings.append(embedding)
               known_labels.append(person_name)

   np.save("known_face_embeddings.npy", np.array(known_embeddings))
   np.save("known_labels.npy", np.array(known_labels))
   ```

4. **Save the Embeddings and Labels:**
   - Save the embeddings and labels to `.npy` files for use during runtime.

---

## **How It Works**

1. The video stream is captured frame by frame.
2. Human detections are identified using YOLO.
3. Face recognition is performed for each detected face.
4. Tracks are updated with matched detections.
5. Known and unknown individuals are counted and displayed on the frame.
6. Frames and logs are saved to the disk for future analysis.

---

## **Running the Script**

1. Modify the configuration parameters as described in [System Configuration](#system-configuration).
2. Run the script:

```bash
python person_tracker.py
```

3. Press `q` to stop the program and release resources.

---

## **Example Output**
- **On-Screen Display:**
  - Counts for known and unknown individuals.
  - FPS (Frames Per Second).
  - Tracked IDs and recognition status.
- **Saved Outputs:**
  - Processed frames with bounding boxes and labels.
  - Logs of detection and recognition events in JSON format.

---

### **Contributions**
Feel free to contribute by opening a pull request or reporting issues in the repository.
