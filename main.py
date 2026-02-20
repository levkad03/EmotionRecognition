import cv2
from src.face_processor_updated import FaceProcessor

processor = FaceProcessor(use_face_mesh=True)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit, 's' to save frame, 'a' to save alignment")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = processor.detect_faces(frame)

    # Process if face detected
    if faces:
        best_face = max(faces, key=lambda x: x["confidence"])

        # Get landmarks
        landmarks = processor.get_face_landmarks(frame, best_face["bbox"])

        if landmarks:
            # Draw landmarks
            display = processor.draw_landmarks(frame.copy(), landmarks)

            # Get aligned face
            aligned = processor.align_faces_advanced(frame, landmarks)

            # Show aligned preview (small)
            h, w = display.shape[:2]
            aligned_small = cv2.resize(aligned, (224 // 2, 224 // 2))
            display[-120:, -120:] = cv2.resize(aligned_small, (120, 120))
        else:
            display = frame
    else:
        display = frame

    cv2.imshow("Face Processor - Webcam Test", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("webcam_frame.jpg", frame)
        print("Frame saved!")

cap.release()
cv2.destroyAllWindows()
