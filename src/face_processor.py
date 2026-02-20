from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceProcessor:
    """Face detection and alignment using InsightFace."""

    def __init__(self, detection_confidence: float = 0.5):
        """Initialize face processor

        Args:
            detection_confidence (float): Minimum confidence for face detection
        """
        self.detection_confidence = detection_confidence

        # Initialize InsightFace app
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect faces in image using InsightFace.

        Args:
            image (np.ndarray): Input image in BGR

        Returns:
            list[dict]: List of dictionaries with face information
        """

        # InsightFace expects RGB

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        detections = self.app.get(rgb_image)

        faces = []

        for face in detections:
            # Skip low confidence detections
            if face.det_score < self.detection_confidence:
                continue

            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Convert to (x, y, width, height) format
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1

            # Get landmarks (5 points) - use 'kps' not 'landmark'
            landmarks = face.kps.astype(int)
            keypoints = {
                "left_eye": landmarks[0],  # Left eye
                "right_eye": landmarks[1],  # Right eye
                "nose": landmarks[2],  # Nose tip
                "mouth_left": landmarks[3],  # Left mouth corner
                "mouth_right": landmarks[4],  # Right mouth corner
            }

            faces.append(
                {
                    "bbox": (x, y, width, height),
                    "confidence": float(face.det_score),
                    "keypoints": keypoints,
                }
            )

        return faces

    def calculate_eye_centers(self, keypoints: dict) -> tuple[np.ndarray, np.ndarray]:
        """Calculate center of left and right eyes from keypoints."""
        if "left_eye" in keypoints and "right_eye" in keypoints:
            left_eye_center = np.array(keypoints["left_eye"])
            right_eye_center = np.array(keypoints["right_eye"])

            return left_eye_center, right_eye_center
        return None, None

    def align_face(
        self,
        image: np.ndarray,
        keypoints: dict,
        output_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Align face using eye positions

        Args:
            image (np.ndarray): Input image.
            keypoints (dict): Facial keypoints.
            output_size (tuple[int, int]): Output image size.

        Returns:
            np.ndarray: Aligned face image.
        """

        left_eye_center, right_eye_center = self.calculate_eye_centers(keypoints)

        if left_eye_center is None or right_eye_center is None:
            return self.crop_face_from_landmarks(image, keypoints, output_size)

        # Calculate angle between eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center point between eyes
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) / 2,
            (left_eye_center[1] + right_eye_center[1]) / 2,
        )

        # Calculate distance between eyes
        eye_distance = np.sqrt(dx**2 + dy**2)

        # Calculate eye positions in output image
        desired_eye_distance = output_size[0] * 0.4
        desired_left_eye = (output_size[0] * 0.3, output_size[1] * 0.35)

        # Calculate scale
        scale = desired_eye_distance / eye_distance

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)

        # Adjust translation
        M[0, 2] += desired_left_eye[0] - left_eye_center[0]
        M[1, 2] += desired_left_eye[1] - left_eye_center[1]

        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            M,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned

    def crop_face_from_landmarks(
        self, image: np.ndarray, keypoints: dict, output_size: tuple[int, int]
    ) -> np.ndarray:
        """Cropp face using landmarks."""

        all_points = [
            keypoints["left_eye"],
            keypoints["right_eye"],
            keypoints["nose"],
            keypoints["mouth_left"],
            keypoints["mouth_right"],
        ]

        points = np.array(all_points)

        # Get bounding box
        x_min, y_min = np.min(points, axis=0).astype(int)
        x_max, y_max = np.max(points, axis=0).astype(int)

        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        padding = int(max(width, height) * 0.3)

        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(image.shape[1], x_max + padding)
        y2 = min(image.shape[0], y_max + padding)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros((*output_size, 3), dtype=np.uint8)

        return cv2.resize(cropped, output_size)

    def process_image(
        self,
        image_path: str,
        output_path: str | None = None,
        align: bool = True,
        visualize: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Process single image: detect, crop and align face."""

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.detect_faces(image)
        if not faces:
            print(f"No faces detected in {image_path}")
            result = cv2.resize(image, (224, 224))
            if output_path:
                cv2.imwrite(output_path, result)

            return result, False

        main_face = max(faces, key=lambda x: x["confidence"])

        if align:
            processed = self.align_face(image, main_face["keypoints"])
        else:
            x, y, w, h = main_face["bbox"]
            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            cropped = image[y1:y2, x1:x2]
            processed = cv2.resize(cropped, (224, 224))

        if visualize and output_path:
            self.draw_landmarks(
                image.copy(),
                main_face["keypoints"],
                output_path.replace(".jpg", "_landmarks.jpg"),
            )

        if output_path:
            cv2.imwrite(output_path, processed)

        return processed, True

    def draw_landmarks(
        self,
        image: np.ndarray,
        keypoints: dict,
        output_path: str | None = None,
    ) -> np.ndarray:
        """Draw facial landmarks on image."""

        img_copy = image.copy()

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
                cv2.circle(img_copy, point, 3, color, -1)

        left_eye = np.array(keypoints["left_eye"])
        right_eye = np.array(keypoints["right_eye"])
        cv2.line(
            img_copy,
            tuple(left_eye.astype(int)),
            tuple(right_eye.astype(int)),
            (255, 255, 0),
            2,
        )

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        cv2.putText(
            img_copy,
            f"Angle: {angle:.1f}°",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if output_path:
            cv2.imwrite(output_path, img_copy)

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        align: bool = True,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPEG", ".PNG"),
    ) -> dict:
        """Process all images in directory.

        Args:
            input_dir (str): Directory with input images.
            output_dir (str): Directory to save processed images.
            align (bool, optional): Whether to align faces. Defaults to True.
            extensions (tuple[str, ...]): File extensions to process.

        Returns:
            dict: A stats dictionary.
        """

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))

        print(f"Found {len(image_files)} images in {input_dir}")

        stats = {"total": len(image_files), "processed": 0, "no_face": 0, "errors": 0}

        for i, img_file in enumerate(image_files):
            try:
                rel_path = img_file.relative_to(input_path)
                output_file = output_path / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                _, face_detected = self.process_image(
                    str(img_file), str(output_file), align=align, visualize=False
                )

                if face_detected:
                    stats["processed"] += 1
                else:
                    stats["no_face"] += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images...")

            except Exception as e:
                stats["errors"] += 1
                print(f"Error processing {img_file}: {str(e)}")

        print("\nProcessing complete:")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  No face detected: {stats['no_face']}")
        print(f"  Errors: {stats['errors']}")

        return stats


if __name__ == "__main__":
    processor = FaceProcessor(detection_confidence=0.5)

    # Test on single image
    aligned_face, detected = processor.process_image(
        "WIN_20260210_12_45_16_Pro.jpg", "output.jpg", align=True, visualize=True
    )
