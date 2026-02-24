import cv2
import numpy as np

from src.face_processor import FaceProcessor


def run_camera_detection():
    """Run face detection and alignment on camera feed."""

    # Initialize face processor
    print("Initializing face processor...")
    processor = FaceProcessor(detection_confidence=0.5)
    print("Face processor ready!")

    # Open camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")

    frame_count = 0

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        frame_count += 1

        # Detect faces in the frame
        faces = processor.detect_faces(frame)

        # Draw bounding boxes and landmarks
        display_frame = frame.copy()

        for face in faces:
            # Draw bounding box
            x, y, w, h = face["bbox"]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw confidence score
            confidence_text = f"{face['confidence']:.2f}"
            cv2.putText(
                display_frame,
                confidence_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw landmarks
            keypoints = face["keypoints"]
            colors = {
                "left_eye": (0, 255, 0),
                "right_eye": (0, 255, 0),
                "nose": (255, 0, 0),
                "mouth_left": (0, 0, 255),
                "mouth_right": (0, 0, 255),
            }

            for key, color in colors.items():
                if key in keypoints:
                    point = tuple(keypoints[key])
                    cv2.circle(display_frame, point, 3, color, -1)

            # Draw line between eyes
            if "left_eye" in keypoints and "right_eye" in keypoints:
                left_eye = tuple(keypoints["left_eye"])
                right_eye = tuple(keypoints["right_eye"])
                cv2.line(display_frame, left_eye, right_eye, (255, 255, 0), 2)

        # Display FPS
        cv2.putText(
            display_frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show the frame
        cv2.imshow("Face Detection - Press Q to quit, S to save", display_frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            # Save current frame
            filename = f"camera_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

            # Save aligned face if detected
            if faces:
                main_face = max(faces, key=lambda x: x["confidence"])
                aligned = processor.align_face(frame, main_face["keypoints"])
                aligned_filename = f"aligned_face_{frame_count}.jpg"
                cv2.imwrite(aligned_filename, aligned)
                print(f"Saved aligned face to {aligned_filename}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")


if __name__ == "__main__":
    run_camera_detection()
