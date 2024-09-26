import insightface
import numpy as np
import cv2
import os
import sqlite3
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import logging
import yaml
from typing import Dict, List, Tuple


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations from a YAML file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Directories and database path from config file
known_faces_dir = config["known_faces_dir"]
group_images_dir = config["group_images_dir"]
db_path = config["db_path"]

# Thresholds from config file
detection_confidence_threshold = config["detection_confidence_threshold"]
matching_threshold = config["matching_threshold"]

def initialize_model() -> insightface.app.FaceAnalysis:
    """
    Initialize the InsightFace model using 'buffalo_l'
    configuration and set confidence threshold.

    Returns:
        model (FaceAnalysis): The initialized face analysis model.
    """
    model = insightface.app.FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=-1)  # ctx_id=-1 forces the use of CPU
    model.detect_conf = detection_confidence_threshold
    return model

def check_known_faces_directory(path: str) -> None:
    """
    Check if the known faces directory is valid and contains subdirectories.

    Args:
        path (str): Path to the known faces directory.

    Raises:
        FileNotFoundError: If the directory does not exist or has no subdirectories.
    """

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")
    if not any(os.path.isdir(os.path.join(path, d))
               for d in os.listdir(path)):
        raise FileNotFoundError(f"No subdirectories found in directory {path}.")

def check_group_images_directory(path: str) -> None:
    """
    Check if the group images directory is valid and contains image files.

    Args:
        path (str): Path to the group images directory.

    Raises:
        FileNotFoundError: If the directory does not exist or has no image files.
    """

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")
    if not any(fname.endswith(('.jpg', '.jpeg', '.png'))
               for fname in os.listdir(path)):
        raise FileNotFoundError(f"No images found in directory {path}.")

def initialize_database(db_path: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Initialize the SQLite database and create necessary tables.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        conn (Connection): SQLite database connection.
        cursor (Cursor): SQLite database cursor.

    Raises:
        sqlite3.Error: If there is an issue with the database connection.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS face_embeddings
                         (label TEXT PRIMARY KEY, embedding BLOB)''')
            c.execute('''CREATE TABLE IF NOT EXISTS group_images
                         (image_path TEXT PRIMARY KEY, labels TEXT, coords TEXT)''')
        return conn, c
    except sqlite3.Error as e:
        logging.error(f"Error initializing database: {e}")
        raise

def prepare_and_save_face_database(known_faces_dir: str,
                                    model: insightface.app.FaceAnalysis,
                                    cursor: sqlite3.Cursor) -> None:
    """
    Extract embeddings for known faces from the directory, calculate average embeddings,
    and store them in the SQLite database.

    Args:
        known_faces_dir (str): Path to the directory of known faces.
        model (FaceAnalysis): Initialized InsightFace model for face analysis.
        cursor (Cursor): SQLite database cursor for saving embeddings.
    """
    face_db: Dict[str, np.ndarray] = {}
    check_known_faces_directory(known_faces_dir)  # Ensure directory is valid

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)

        if os.path.isdir(person_dir):
            embeddings = []

            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(person_dir, filename)
                    img = cv2.imread(filepath)

                    if img is None:
                        logging.error(f"Image at path {filepath} could not be loaded.")
                        continue
                    faces = model.get(img)

                    if faces:
                        face = faces[0]
                        embeddings.append(face.normed_embedding)

            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)  # Average embedding for robustness
                face_db[person_name] = avg_embedding  # Save the average embedding with the folder name as the label
    save_to_sqlite(face_db, cursor)

def save_to_sqlite(face_db: Dict[str, np.ndarray], cursor: sqlite3.Cursor) -> None:
    """
    Save face embeddings to the SQLite database.

    Args:
        face_db (Dict[str, np.ndarray]): Dictionary of face labels and their embeddings.
        cursor (Cursor): SQLite database cursor.
    """
    for label, embedding in face_db.items():
        embedding_blob = embedding.tobytes()
        cursor.execute("INSERT OR REPLACE INTO face_embeddings (label, embedding) VALUES (?, ?)",
                       (label, embedding_blob))

def load_from_sqlite(cursor: sqlite3.Cursor) -> Dict[str, np.ndarray]:
    """
    Load face embeddings from the SQLite database.

    Args:
        cursor (Cursor): SQLite database cursor.

    Returns:
        face_db (Dict[str, np.ndarray]): Dictionary of face labels and their embeddings.
    """
    cursor.execute("SELECT label, embedding FROM face_embeddings")
    face_db = {}

    for row in cursor.fetchall():
        label, embedding_blob = row
        embedding = np.frombuffer(
            embedding_blob, dtype=np.float32)
        face_db[label] = embedding
    return face_db

def process_group_images(group_images_dir: str, model: insightface.app.FaceAnalysis,
                          face_db: Dict[str, np.ndarray], cursor: sqlite3.Cursor,
                            threshold: float) -> None:
    """
    Process group images, detect faces, compare embeddings with known faces, and store the results in SQLite.

    Args:
        group_images_dir (str): Path to the directory of group images.
        model (FaceAnalysis): Initialized InsightFace model for face analysis.
        face_db (Dict[str, np.ndarray]): Dictionary of known face embeddings.
        cursor (Cursor): SQLite database cursor for saving group image data.
        threshold (float): Cosine similarity threshold for face matching.
    """
    check_group_images_directory(group_images_dir)  # Ensure directory is valid
    for filename in os.listdir(group_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(group_images_dir, filename)
            img = cv2.imread(filepath)

            if img is None:
                logging.error(f"Image at path {filepath} could not be loaded.")
                continue
            faces = model.get(img)
            labels = []
            coords = []

            for face in faces:
                embedding = face.normed_embedding
                min_dist = float('inf')
                label = 'Unknown'

                for name, db_embedding in face_db.items():
                    dist = cosine(embedding, db_embedding)

                    if dist < min_dist:
                        min_dist = dist
                        label = name if dist < threshold else 'Unknown'
                labels.append(label)
                bbox = face.bbox.astype(int)
                coords.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                               (0, 255, 0), 2)
                cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if labels:
                cursor.execute("INSERT OR REPLACE INTO group_images \
                               (image_path, labels, coords) VALUES (?, ?, ?)",
                               (filepath, ','.join(labels),
                                ';'.join(coords)))

def retrieve_images_by_labels(required_labels: List[str], excluded_labels: List[str],
                               cursor: sqlite3.Cursor) -> List[Tuple[str, str, str]]:
    """
    Retrieve images from the SQLite database that match the required labels and\
          exclude the specified labels.

    Args:
        required_labels (List[str]): List of labels that must be present in the images.
        excluded_labels (List[str]): List of labels that must not be present in the images.
        cursor (Cursor): SQLite database cursor.

    Returns:
        List[Tuple[str, str, str]]: List of matching images with their file paths, coordinates, and labels.
    """
    required_labels_set = set(required_labels)
    excluded_labels_set = set(excluded_labels)
    cursor.execute("SELECT image_path, labels, coords FROM group_images")
    results = []
    for row in cursor.fetchall():
        image_path, labels, coords = row
        labels_set = set(labels.split(','))
        if required_labels_set.issubset(labels_set) and excluded_labels_set.isdisjoint(labels_set):
            results.append((image_path, coords, labels))
    return results

def display_image_with_boxes(image_path: str, coords: List[str], labels: List[str]) -> None:
    """
    Display the image with bounding boxes around the detected faces and their corresponding labels.

    Args:
        image_path (str): Path to the image file.
        coords (List[str]): List of bounding box coordinates for faces in the format "x1,y1,x2,y2".
        labels (List[str]): List of labels corresponding to each bounding box.
    """
    img = cv2.imread(image_path)

    if img is None:
        logging.error(f"Failed to load image at path: {image_path}")
        return

    for coord, label in zip(coords, labels):
        x1, y1, x2, y2 = map(int, coord.split(','))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to run the entire process
def main(known_faces_dir, group_images_dir, db_path, threshold):
    # Initialize model
    model = initialize_model()

    # Initialize SQLite database
    conn, cursor = initialize_database(db_path)

    # Prepare and save known faces embeddings into database
    prepare_and_save_face_database(known_faces_dir, model, cursor)

    # Load face embeddings from database
    face_db = load_from_sqlite(cursor)

    # Process group images and store results in database
    process_group_images(group_images_dir, model,
                        face_db, cursor, matching_threshold)

    # Example of retrieving images by required and excluded labels
    required_labels = ['Abdolrazzaghi-Elika']
    excluded_labels = ['']
    matching_images = retrieve_images_by_labels(
        required_labels, excluded_labels, cursor)

    for image_path, coords, labels in matching_images:
        display_image_with_boxes(image_path, coords.split(';'),
                                  labels.split(','))

    conn.commit()
    conn.close()

# Run the main function
if __name__ == "__main__":
    main(known_faces_dir, group_images_dir, db_path, matching_threshold)

