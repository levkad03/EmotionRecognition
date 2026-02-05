from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class FaceProcessor:
    """Face detection and algnment using MediaPipe."""

    def __init__(
        self,
        detection_model_selection: int = 1,
        detection_confidence: float = 0.5,
        use_face_mesh: bool = True,
        mesh_confidence: float = 0.5,
    ):
        """Initialize face processor

        Args:
            detection_model_selection (int): 0 for short range, 1 for full range.
            detection_confidence (float): Minimum confidence for face detection.
            use_face_mesh (bool): Whether to use facial landmars for alignment.
            mesh_confidence (float): Minimum confidence for face mesh.
        """

        self.detection_model_selection = detection_model_selection
        self.detection_confidence = detection_confidence
        self.use_face_mesh = use_face_mesh

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=detection_model_selection,
            min_detection_confidence=detection_confidence,
        )

        if use_face_mesh:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,  # Get iris landmarks for better alignment
                min_detection_confidence=mesh_confidence,
                min_tracking_confidence=mesh_confidence,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            # Facial landmark indices for MediaPipe Face Mesh (468 points)
            # Key points for alignment: eyes, nose, mouth corners
            self.LANDMARK_INDICES = {
                "left_eye": [33, 133, 157, 158, 159, 160, 161, 173],  # Left eye contour
                "right_eye": [
                    362,
                    263,
                    386,
                    387,
                    388,
                    389,
                    390,
                    466,
                ],  # Right eye contour
                "nose_tip": 1,  # Nose tip
                "mouth_left": 61,  # Left mouth corner
                "mouth_right": 291,  # Right mouth corner
                "face_oval": [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ],  # Face oval
            }

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect faces in image using MediaPipe.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            list[dict]: List of dictionaries with face information.
        """

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False

        # Perform face detection
        results = self.face_detection.process(rgb_image)

        faces = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                # Convert to pixel coordinates
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Get keypoints if available
                keypoints = {}

                if detection.location_data.relative_keypoints:
                    keypoints = {
                        "right_eye": detection.location_data.relative_keypoints[0],
                        "left_eye": detection.location_data.relative_keypoints[1],
                        "nose_tip": detection.location_data.relative_keypoints[2],
                        "mouth_center": detection.location_data.relative_keypoints[3],
                        "right_ear": detection.location_data.relative_keypoints[4],
                        "left_ear": detection.location_data.relative_keypoints[5],
                    }

                faces.append(
                    {
                        "bbox": (x1, y1, width, height),
                        "confidence": detection.score[0],
                        "keypoints": keypoints,
                    }
                )

        return faces

    def get_face_landmarks(
        self, image: np.ndarray, bbox: tuple | None = None
    ) -> dict | None:
        """Get detailed facial landmarks using MediaPipe Face Mesh

        Args:
            image (np.ndarray): Input image (BGR format).
            bbox (tuple | None): Bounding box to crop face (x, y, w, h).

        Returns:
            dict | None: Dictionary with facial landmarks or None if no face detected.
        """

        if not self.use_face_mesh:
            return None

        # If bbox provided, crop face region for more accurate landmark detection
        if bbox is not None:
            x, y, w, h = bbox
            # Add padding to ensure we capture entire face
            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            face_region = image[y1:y2, x1:x2]
        else:
            face_region = image

        # Convert to RGB
        rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        rgb_face.flags.writeable = False

        # Get face mesh
        results = self.face_mesh.process(rgb_face)

        if results.multi_face_landmarks:
            # Get the first face
            # (assuming single face per image for emotion recognition)
            face_landmarks = results.multi_face_landmarks[0]

            # Convert landmarks to pixel coordinates
            landmarks = {}
            h, w = face_region.shape[:2]

            # Get all 468 landmarks
            all_points = []
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Adjust coordinates if we cropped the image
                if bbox is not None:
                    px = int((landmark.x * w + x1) * w)
                    py = int((landmark.y * h + y1) * h)
                else:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)

                all_points.append((px, py))

                # Store specific key landmarks
                if idx in [33, 133, 157, 158, 159, 160, 161, 173]:  # Left eye
                    landmarks.setdefault("left_eye", []).append((px, py))
                elif idx in [362, 263, 386, 387, 388, 389, 390, 466]:  # Right eye
                    landmarks.setdefault("right_eye", []).append((px, py))
                elif idx == 1:  # Nose tip
                    landmarks["nose_tip"] = (px, py)
                elif idx == 61:  # Left mouth corner
                    landmarks["mouth_left"] = (px, py)
                elif idx == 291:  # Right mouth corner
                    landmarks["mouth_right"] = (px, py)

            landmarks["all_landmarks"] = all_points
            return landmarks

        return None

    def calculate_eye_centers(self, landmarks: dict) -> tuple[np.ndarray, np.ndarray]:
        """Calculate center of left and right eyes from landmarks."""

        if "left_eye" in landmarks and "right_eye" in landmarks:
            left_eye_points = np.array(landmarks["left_eye"])
            right_eye_points = np.array(landmarks["right_eye"])

            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)

            return left_eye_center, right_eye_center

        return None, None

    def align_faces_advanced(
        self,
        image: np.ndarray,
        landmarks: dict,
        output_size: tuple[int, int] = (224, 224),
        scale_factor: float = 1.5,
    ) -> np.ndarray:
        """Advanced face alignment using MediaPipe landmarks.

        Args:
            image (np.ndarray): Input image.
            landmarks (dict): Facial landmarks from MediaPipe.
            output_size (tuple[int, int]): Output image size.
            scale_factor (float): How much to scale the face region.

        Returns:
            np.ndarray: Aligned face image.
        """

        # Calculate eye centers
        left_eye_center, right_eye_center = self.calculate_eye_centers(landmarks)

        if left_eye_center is None or right_eye_center is None:
            # Fallback to simple cropping if eye landmarks not available
            return self.crop_face_smart(image, landmarks, output_size)

        # Calculate angle between eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center point between eyes
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) // 2,
            (left_eye_center[1] + right_eye_center[1]) // 2,
        )

        # Calculate distance between eyes
        eye_distance = np.sqrt(dx**2 + dy**2)

        # Calculate desired eye positions in output image (for horizontal alignment)
        desired_eye_distance = output_size[0] * 0.3  # Eyes at 30% and 70% of width
        desired_left_eye = (output_size[0] * 0.3, output_size[1] * 0.35)

        # Calculate scale
        scale = desired_eye_distance / eye_distance * scale_factor

        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Adjust translation to position eyes correctly
        rotation_matrix[0, 2] += desired_left_eye[0] - left_eye_center[0]
        rotation_matrix[1, 2] += desired_left_eye[1] - left_eye_center[1]

        # Apply affine transformation
        aligned_face = cv2.warpAffine(
            image,
            rotation_matrix,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned_face

    def crop_face_smart(
        self,
        image: np.ndarray,
        landmarks: dict,
        output_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Smart cropping using facial landmarks to ensure entire face is captured.

        Args:
            image (np.ndarray): Input image.
            landmarks (dict): Facial landmarks.
            output_size (tuple[int, int]): Output image size.

        Returns:
            np.ndarray: Cropped face image.
        """

        if "all_points" not in landmarks:
            # Fallback to bounding box based cropping
            all_points = []

            for key, val in landmarks.items():
                if key in ["left_eye", "right_eye"]:
                    all_points.extend(val)
                elif key in ["nose_tip", "mouth_left", "mouth_right"]:
                    all_points.append(val)

            if not all_points:
                # If no landmarks, return center crop
                h, w = image.shape[:2]
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                cropped = image[
                    start_h : start_h + min_dim, start_w : start_w + min_dim
                ]
                return cv2.resize(cropped, output_size)

            points = np.array(all_points)
        else:
            points = np.array(landmarks["all_points"])

        # Get bounding box from landmarks with padding
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Add padding (20% of face width/height)
        width = x_max - x_min
        height = y_max - y_min

        padding_w = int(width * 0.2)
        padding_h = int(height * 0.2)

        # Ensure coordinates are within image bounds
        x1 = max(0, int(x_min - padding_w))
        y1 = max(0, int(y_min - padding_h))
        x2 = min(image.shape[1], int(x_max + padding_w))
        y2 = min(image.shape[0], int(y_max + padding_h))

        # Crop and resize
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
    ) -> np.ndarray:
        """Process single image: detect, crop and align face

        Args:
            image_path (str): Path to input image.
            output_path (str | None): Optional path to save processed image.
            align (bool): Whether to align face using landmarks.
            visualize (bool): Whether to draw landmarks on output.

        Returns:
            np.ndarray: Processed image.
        """

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Detect faces
        faces = self.detect_faces(image)
        if not faces:
            print(f"No faces detected in {image_path}")
            # Return resized image if no face detected
            result = cv2.resize(image, (224, 224))
            if output_path:
                cv2.imwrite(output_path, result)

            return result

        # Use the face with highest confidence
        main_face = max(faces, key=lambda x: x["confidence"])

        if align and self.use_face_mesh:
            # Get detailed landmarks
            landmarks = self.get_face_landmarks(image, main_face["bbox"])

            if landmarks:
                # Use advanced alignment
                processed = self.align_faces_advanced(image, landmarks)

                # Draw landmarks if visualization is enabled
                if visualize and output_path:
                    self.draw_landmarks(
                        image.copy(),
                        landmarks,
                        output_path.replace(".jpg", "_landmarks.jpg"),
                    )
            else:
                # Fallback to smart cropping
                processed = self.crop_face_smart(image, main_face["keypoints"])

        else:
            # Simple cropping using bounding box
            x, y, w, h = main_face["bbox"]
            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            cropped = image[y1:y2, x1:x2]
            processed = cv2.resize(cropped, (224, 224))

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, processed)

        return processed

    def draw_landmarks(
        self, image: np.ndarray, landmarks: dict, output_path: str | None = None
    ) -> np.ndarray:
        """Draw facial landmarks on image."""

        img_copy = image.copy()

        # Draw eye landmarks
        if "left_eye" in landmarks:
            for point in landmarks["left_eye"]:
                cv2.circle(img_copy, tuple(point), 2, (0, 255, 0), -1)

        if "right_eye" in landmarks:
            for point in landmarks["right_eye"]:
                cv2.circle(img_copy, tuple(point), 2, (0, 255, 0), -1)

        # Draw other key points
        colors = {
            "nose_tip": (255, 0, 0),
            "mouth_left": (0, 0, 255),
            "mouth_right": (0, 0, 255),
        }

        for key, color in colors.items():
            if key in landmarks:
                cv2.circle(img_copy, landmarks[key], 3, color, -1)

        # Draw eye line for visualization
        left_eye_center, right_eye_center = self.calculate_eye_centers(landmarks)
        if left_eye_center is not None and right_eye_center is not None:
            cv2.line(
                img_copy,
                tuple(left_eye_center.astype(int)),
                tuple(right_eye_center.astype(int)),
                (255, 255, 0),
                2,
            )

            # Draw angle text
            dx = right_eye_center[0] - left_eye_center[0]
            dy = right_eye_center[1] - left_eye_center[1]

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

        return img_copy

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        align: bool = True,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    ):
        """Process all images in the directory.

        Args:
            input_dir (str): Input directory with images.
            output_dir (str): Output directory for processed images.
            align (bool): Whether to align faces.
            extensions (tuple): Image file extensions to process.
        """

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))

        print(f"Found {len(image_files)} images in {input_dir}")

        stats = {"total": len(image_files), "processed": 0, "no_face": 0, "errors": 0}

        for i, img_file in enumerate(image_files):
            try:
                # Create relative output path
                rel_path = img_file.relative_to(input_path)
                output_file = output_path / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Process image
                self.process_image(
                    str(img_file), str(output_file), align=align, visualize=False
                )

                stats["processed"] += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")

            except Exception as e:
                stats["errors"] += 1
                print(f"Error processing {img_file}: {str(e)}")

        print("\nProcessing complete:")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  No face detected: {stats['no_face']}")
        print(f"  Errors: {stats['errors']}")

        return stats
