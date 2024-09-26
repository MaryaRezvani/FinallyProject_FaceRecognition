import insightface
import cv2
import os
from scipy.spatial.distance import cosine


# Initialize the InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # Used CPU

# Function to extract embeddings for known faces
def prepare_face_database(known_faces_dir):
    face_db = {}
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            img = cv2.imread(filepath)
            faces = model.get(img)
            if faces:
                face = faces[0]  # Assume one face per image
                embedding = face.normed_embedding
                label = os.path.splitext(filename)[0]
                face_db[label] = embedding
    return face_db

# Directory containing images of known faces
known_faces_dir = ''
# Prepare the face database
face_db = prepare_face_database(known_faces_dir)

# Function to recognize faces in an image and label them
def recognize_and_label_faces(image_path, face_db):
    img = cv2.imread(image_path)
    faces = model.get(img)
    img_height, img_width, _ = img.shape

    for face in faces:
        embedding = face.normed_embedding
        min_dist = float('inf')
        label = 'Unknown'

        # Compare embedding with the face database
        for name, db_embedding in face_db.items():
            dist = cosine(embedding, db_embedding)
            if dist < min_dist:
                min_dist = dist
                label = name if dist < 0.7 else 'Unknown'  # Threshold for recognition

        # Draw bounding box on the image
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Calculate position and size for the label text
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = bbox[0]
        text_y = bbox[3] + text_size[1] + 10  # Position the text below the bounding box

        # Ensure the text stays within image bounds
        if text_y > img_height:
            text_y = img_height - 10  # Adjust text position if it goes beyond image height

        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image in a resizable window
    cv2.namedWindow('Recognized Faces', cv2.WINDOW_NORMAL)  # Allow window to be resizable
    cv2.imshow('Recognized Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = ''
recognize_and_label_faces(image_path, face_db)
