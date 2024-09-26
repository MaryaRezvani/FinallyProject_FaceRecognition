import insightface
import cv2

# Load the InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # Used CPU

# Function to recognize faces in an image
def recognize_faces(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Detect faces
    faces = model.get(img)
    # print(faces)

    # Draw bounding boxes and landmarks on the image
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2, )
        # Add confidence score
        text = f'{face.det_score:.2f}'
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, f'{face.det_score:.2f}', (bbox[0], bbox[3] + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Create a resizable window
    cv2.namedWindow('Recognized Faces', cv2.WINDOW_NORMAL)

    # Display the image
    cv2.imshow('Recognized Faces', img)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
recognize_faces('')
