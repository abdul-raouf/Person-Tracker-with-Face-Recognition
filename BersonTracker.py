import os
import cv2
import time
import json
import torch
import numpy as np
from datetime import datetime
from time import time as get_current_time
from threading import Thread
from queue import Queue

# Scipy libraries to compare face embeddings with face detections
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# InsightFace FaceDetector and FaceRecognition
import insightface
from insightface.app import FaceAnalysis

# YOLO import
from ultralytics import YOLO

# Configuration Parameters
YOLO_MODEL_PATH = r"D:\FCR\yolo11s.pt"
KNOWN_EMBEDDINGS_PATH = r"D:\FCR\FaceRecognition\known_face_embeddings.npy"
KNOWN_LABELS_PATH = r"D:\FCR\FaceRecognition\known_labels.npy"
DATA_SAVE_DIR = r"D:\FCR\FaceRecognition\PersonTrackerData"
STREAM_URL = r""

MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Initialize models and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv8-face model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Initialize ArcFace's FaceAnalysis model with face detection and embedding capabilities
face_recognition_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_recognition_model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, -1 for CPU

# Load embeddings and labels for known faces
known_embeddings = np.load(KNOWN_EMBEDDINGS_PATH)
known_labels = np.load(KNOWN_LABELS_PATH)

# Paths for JSON and frame saving
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(DATA_SAVE_DIR, f"person_details_{timestamp_str}")
json_file_path = os.path.join(run_dir, "detection_log.json")
frames_dir = os.path.join(run_dir, "saved_frames")

os.makedirs(frames_dir, exist_ok=True)

# Counts for logging
known_count = 0
unknown_count = 0
last_known_count = 0
last_unknown_count = 0

fps = 0
frame_count = 0
fps_start_time = get_current_time()

# Face Similarity Threshold
SIMILARITY_THRESHOLD = 0.6

# Thread-safe queues for logging and frame saving
log_queue = Queue()
frame_queue = Queue()

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.name_confirmed = False
        self.name = "Unknown"
        self.recognition_counter = 0
        self.missing_counter = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.embedding = None
        self.frames_since_recognition = 0  # For recognition interval

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.time_since_update = 0
        self.missing_counter = 0
        self.hit_streak += 1

    def increment_counters(self):
        self.missing_counter += 1
        self.time_since_update += 1
        self.hit_streak = 0

class TrackManager:
    def __init__(self, max_missing=20, iou_threshold=0.35, recognition_interval=2):
        self.tracks = []
        self.next_id = 0
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.recognition_interval = recognition_interval

    def _calculate_iou(self, box1, box2):
        # Compute intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        # Compute union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def update_tracks(self, detections, frame):
        global known_count, unknown_count

        cost_matrix = []
        for track in self.tracks:
            row = [1 - self._calculate_iou(track.bbox, det) for det in detections]
            cost_matrix.append(row)

        if cost_matrix and detections:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = [(row, col) for row, col in zip(row_ind, col_ind)
                       if cost_matrix[row][col] < 1 - self.iou_threshold]
        else:
            matches = []

        matched_tracks = set()
        matched_detections = set()
        RECOGNITION_INTERVAL = 2

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            track.update(detection)
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

            if not track.name_confirmed:
                track.frames_since_recognition += 1
                if track.frames_since_recognition >= RECOGNITION_INTERVAL:
                    label_matched = self.recognize_face(detection, frame)
                    if label_matched != "Unknown":
                        if label_matched == track.name:
                            track.recognition_counter += 1
                        else:
                            track.recognition_counter = 1
                            track.name = label_matched
                    else:
                        # Decay recognition counter if face not recognized
                        track.recognition_counter = max(0, track.recognition_counter - 1)
                        track.name = "Unknown"

                    if track.recognition_counter > 10:
                        track.name_confirmed = True
                        known_count += 1  # Count as known person
                    track.frames_since_recognition = 0  # Reset counter

        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                new_track = Track(detection, self.next_id)
                self.next_id += 1
                label_matched = self.recognize_face(detection, frame)
                if label_matched != "Unknown":
                    new_track.recognition_counter = 1
                    new_track.name = label_matched
                if new_track.recognition_counter > 10:
                    new_track.name_confirmed = True
                    known_count += 1
                else:
                    unknown_count += 1  # Count as unknown if not recognized
                self.tracks.append(new_track)

        # Increment missing counters for unmatched tracks
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.increment_counters()

        # Remove tracks missing for too long
        self.tracks = [t for t in self.tracks if t.missing_counter <= self.max_missing]

    def recognize_face(self, bbox, frame):
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        person_roi = frame[y1:y2, x1:x2]

        if person_roi.size == 0 or person_roi.shape[0] == 0 or person_roi.shape[1] == 0:
            return "Unknown"
        try:
            faces = face_recognition_model.get(person_roi)
            
        except Exception as e:
            print(f"Error during face recognition: {e}")
            return "Unknown"
        
        if faces:
            face = faces[0]
            embedding = face.embedding
            label, _ = find_closest_match(embedding)
            return label
        else:
            return "Unknown"

    def get_tracks(self):
        return self.tracks

def find_closest_match(embedding):
    if embedding is None or len(known_embeddings) == 0:
        return "Unknown", 0.0

    similarities = [1 - cosine(embedding, known_emb) for known_emb in known_embeddings]
    max_similarity = max(similarities)
    max_index = np.argmax(similarities)

    if max_similarity > SIMILARITY_THRESHOLD:
        return known_labels[max_index], max_similarity
    return "Unknown", max_similarity

def log_writer():
    while True:
        log_entry = log_queue.get()
        if log_entry is None:
            log_queue.task_done()
            print("log_writer received exit signal.")
            break
        print("log_writer processing log entry.")
        with open(json_file_path, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        log_queue.task_done()

def frame_saver():
    while True:
        frame_info = frame_queue.get()
        if frame_info is None:
            frame_queue.task_done()
            print("frame_saver received exit signal.")
            break
        print("frame_saver processing frame.")
        frame, frame_path = frame_info
        cv2.imwrite(frame_path, frame)
        frame_queue.task_done()


## Call this function to generate face embeddings for new person.
## **PARAMS** -> Input directory path which contains images of the person.
def generate_embeddings(base_dir):
    """Generate and save face embeddings from a directory of face images using ArcFace."""
    embeddings = []
    labels = []
    
    print("Starting training phase...")
    
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        person_embeddings = []
        valid_faces = 0
        print(f"\nProcessing images for {person_name}...")
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            try:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not load image: {image_path}")
                    continue

                # Use ArcFace's get method to detect faces and obtain embeddings
                faces = face_recognition_model.get(img)

                # Ensure at least one face is detected
                if faces:
                    embedding = faces[0].embedding  # Get embedding for the first face detected
                    person_embeddings.append(embedding)
                    valid_faces += 1
                    print(f"Successfully processed {image_name}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Only add the person if we have enough valid embeddings
        if len(person_embeddings) >= 3:
            mean_embedding = np.mean(person_embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)  # Normalize embedding
            embeddings.append(mean_embedding)
            labels.append(person_name)
            print(f"Added {person_name} with {valid_faces} valid face images")
        else:
            print(f"Skipped {person_name}: insufficient valid faces ({valid_faces})")
    
    # Save embeddings and labels if they exist
    if embeddings:
        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels)
        print("Embeddings Array:", embeddings_array)
        print("Labels Array:", labels_array)
        
        try:
            np.save(r"D:\FCR\FaceRecognition\known_face_embeddings.npy", embeddings_array)
            np.save(r"D:\FCR\FaceRecognition\known_labels.npy", labels_array)
        except Exception as e:
            print("Error saving files:", str(e))
        print("\nTraining phase completed.")
        print(f"Generated embeddings for {len(labels)} people")
    else:
        print("No valid embeddings generated!")




def main():
    global known_count, unknown_count, last_known_count, last_unknown_count
    track_manager = TrackManager()

    fps = 0
    frame_count = 0
    fps_start_time = get_current_time()
    retries = 0

    """ UNCOMMENT TO GENERATE NEW FACE EMBEDDINGS """
    # generate_embeddings(r"D:\FCR\FaceRecognition\Dataset")

    # Start logging and frame saving threads
    log_thread = Thread(target=log_writer)
    frame_thread = Thread(target=frame_saver)
    log_thread.start()
    frame_thread.start()

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    else:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Initialize timing variables
    last_count_change_time = None
    BUFFER_DURATION = 2.0  # Duration in seconds

    try:
        while True:
            
            ret, frame = cap.read()
            if not ret:
                retries += 1
                if retries > MAX_RETRIES:
                    print("Error: Maximum retries reached. Exiting.")
                    break
                print(f"Warning: Frame read failed. Retrying {retries}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY)
                cap.release()
                cap = cv2.VideoCapture(STREAM_URL)
                continue
            else:
                retries = 0

            frame = cv2.resize(frame, (1280, 1080), interpolation=cv2.INTER_AREA)
            # YOLO detection for human boxes
            yolo_results = yolo_model(frame, classes=[0], conf=0.5, iou=0.5)
            detections = []
            for result in yolo_results:
                for box in result.boxes.xyxy.cpu().numpy().astype(int):
                    detections.append(box)

            # Update tracks with detections
            track_manager.update_tracks(detections, frame)
            active_tracks = track_manager.get_tracks()

            # Count known and unknown people
            current_known_names = {track.name for track in active_tracks if track.name_confirmed and track.name != "Unknown"}
            current_known_count = len(current_known_names)
            current_unknown_count = sum(1 for track in active_tracks if not track.name_confirmed or track.name == "Unknown")

            current_time = get_current_time()

            # Check if counts have changed
            counts_changed = (current_known_count != last_known_count or
                              current_unknown_count != last_unknown_count)

            if counts_changed:
                # Counts have changed, reset the timer
                last_count_change_time = current_time
                last_known_count = current_known_count
                last_unknown_count = current_unknown_count
            else:
                # Counts have not changed, check if buffer duration has passed
                if last_count_change_time is not None:
                    elapsed_time = current_time - last_count_change_time
                    if elapsed_time >= BUFFER_DURATION:
                        # Buffer duration has passed with stable counts

                                    # Draw counts on the frame before saving
                        cv2.putText(frame, f"Known: {current_known_count}", (frame.shape[1] - 300, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"Unknown: {current_unknown_count}", (frame.shape[1] - 300, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.putText(frame, f"Total Count: {current_unknown_count +current_known_count }", (frame.shape[1] - 300, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 180), 2)
                        
                        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 300, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


                        # Draw tracked IDs and labels on the frame before saving
                        for track in active_tracks:
                            x1, y1, x2, y2 = track.bbox
                            if track.name_confirmed:
                                label = track.name
                                color = (0, 255, 0)  # Green for confirmed names
                            else:
                                label = "Recognizing..." if track.recognition_counter > 0 else "Unknown"
                                color = (0, 255, 255) if track.recognition_counter > 0 else (0, 0, 255)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"ID:{track.track_id} - {label}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Log entry
                        log_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "known_people": list(current_known_names),
                            "known_count": current_known_count,
                            "unknown_count": current_unknown_count
                        }
                        log_queue.put(log_entry)

                        # Save the current frame with bounding boxes and labels
                        frame_path = os.path.join(frames_dir, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        frame_queue.put((frame.copy(), frame_path))

                        # Reset the timer to avoid repeated logging
                        last_count_change_time = None

            # Draw tracked IDs and labels on the live frame
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox
                if track.name_confirmed:
                    label = track.name
                    color = (0, 255, 0)  # Green for confirmed names
                else:
                    label = "Recognizing..." if track.recognition_counter > 0 else "Unknown"
                    color = (0, 255, 255) if track.recognition_counter > 0 else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track.track_id} - {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FPS and performance display
            frame_count += 1
            elapsed_time_fps = get_current_time() - fps_start_time
            if elapsed_time_fps >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time_fps
                frame_count = 0
                fps_start_time = get_current_time()

            cv2.putText(frame, f"Known: {current_known_count}", (frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Unknown: {current_unknown_count}", (frame.shape[1] - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Total Count: {current_unknown_count +current_known_count }", (frame.shape[1] - 300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 180), 2)
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 300, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            # Display the frame with bounding boxes and labels
            cv2.imshow("Tracking and Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Signal threads to exit
        log_queue.put(None)
        frame_queue.put(None)
        # Wait for queues to be processed
        log_queue.join()
        frame_queue.join()
        # Wait for threads to finish
        log_thread.join()
        frame_thread.join()
        print("Resources released.")


if __name__ == "__main__":
    main()
