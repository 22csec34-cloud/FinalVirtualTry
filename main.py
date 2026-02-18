# ----------- IMPORTS -----------
from rembg import remove
from PIL import Image
import io
import os
import cv2
import numpy as np
import mediapipe as mp


# ----------- FOLDERS -----------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TRANSPARENT_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "image1.png")

# ----------- BACKGROUND REMOVAL FUNCTION -----------
def remove_background_and_save(pil_image, save_path=TRANSPARENT_IMAGE_PATH):
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    result = remove(img_bytes.getvalue())
    transparent_image = Image.open(io.BytesIO(result)).convert("RGBA")
    transparent_image.save(save_path)
    return transparent_image

# ----------- VIRTUAL TRY-ON CLASS -----------
class VirtualTryOn:
    def __init__(self, cloth_path=TRANSPARENT_IMAGE_PATH):
        self.cloth_path = cloth_path
        
        # MediaPipe Pose (working approach from your reference code)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load cloth
        self.cloth = cv2.imread(self.cloth_path, cv2.IMREAD_UNCHANGED)
        if self.cloth is None:
            raise ValueError(f"Cannot load cloth image from {self.cloth_path}")
        self.cloth_keypoints = self._detect_cloth_keypoints(self.cloth)

    def _detect_cloth_keypoints(self, cloth):
        if cloth.shape[2] == 4:
            alpha = cloth[:, :, 3]
        else:
            alpha = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in cloth image")
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        keypoints = {
            'left_shoulder': (x + int(w*0.2), y + int(h*0.1)),
            'right_shoulder': (x + int(w*0.8), y + int(h*0.1)),
            'left_hem': (x + int(w*0.2), y + h),
            'right_hem': (x + int(w*0.8), y + h)
        }
        return keypoints

    def warp_cloth(self, body_landmarks, frame):
        cloth_shoulder_width = np.linalg.norm(
            np.array(self.cloth_keypoints['left_shoulder']) - np.array(self.cloth_keypoints['right_shoulder'])
        )
        body_shoulder_width = np.linalg.norm(
            np.array(body_landmarks['left_shoulder']) - np.array(body_landmarks['right_shoulder'])
        )
        scale_factor = body_shoulder_width / cloth_shoulder_width

        new_width = int(self.cloth.shape[1] * scale_factor)
        new_height = int(self.cloth.shape[0] * scale_factor)
        cloth_resized = cv2.resize(self.cloth, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        src_points = np.float32([
            [self.cloth_keypoints['left_shoulder'][0]*scale_factor, self.cloth_keypoints['left_shoulder'][1]*scale_factor],
            [self.cloth_keypoints['right_shoulder'][0]*scale_factor, self.cloth_keypoints['right_shoulder'][1]*scale_factor],
            [self.cloth_keypoints['left_hem'][0]*scale_factor, self.cloth_keypoints['left_hem'][1]*scale_factor],
            [self.cloth_keypoints['right_hem'][0]*scale_factor, self.cloth_keypoints['right_hem'][1]*scale_factor]
        ])

        dst_points = np.float32([
            body_landmarks['left_shoulder'],
            body_landmarks['right_shoulder'],
            [body_landmarks['left_hip'][0], body_landmarks['left_hip'][1]],
            [body_landmarks['right_hip'][0], body_landmarks['right_hip'][1]]
        ])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        output_h, output_w = frame.shape[:2]
        warped_cloth = cv2.warpPerspective(cloth_resized, M, (output_w, output_h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return warped_cloth

    def blend(self, frame, warped_cloth):
        if warped_cloth.shape[2] == 4:
            alpha = warped_cloth[:, :, 3] / 255.0
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            cloth_rgb = warped_cloth[:, :, :3]
            blended = (alpha * cloth_rgb + (1 - alpha) * frame).astype(np.uint8)
            return blended
        else:
            return frame

    def estimate_size(self, shoulder_width):
        """Estimate T-shirt size based on shoulder width (pixels, approximate)"""
        if shoulder_width < 100:
            return "S"
        elif shoulder_width < 130:
            return "M"
        elif shoulder_width < 160:
            return "L"
        elif shoulder_width < 190:
            return "XL"
        elif shoulder_width < 220:
            return "XXL"
        else:
            return "XXXL"

    def draw_body_landmarks(self, frame, body_landmarks):
        # Red dots for key points
        for pt in body_landmarks.values():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        # Green lines for body structure
        cv2.line(frame, body_landmarks['left_shoulder'], body_landmarks['right_shoulder'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['left_hip'], body_landmarks['right_hip'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['left_shoulder'], body_landmarks['left_hip'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['right_shoulder'], body_landmarks['right_hip'], (144, 238, 144), 2)

        # Calculate measurements
        shoulder_width = np.linalg.norm(
            np.array(body_landmarks['left_shoulder']) - np.array(body_landmarks['right_shoulder'])
        )
        torso_height = np.linalg.norm(
            np.array([
                (body_landmarks['left_shoulder'][0] + body_landmarks['right_shoulder'][0]) // 2,
                (body_landmarks['left_shoulder'][1] + body_landmarks['right_shoulder'][1]) // 2
            ]) - np.array([
                (body_landmarks['left_hip'][0] + body_landmarks['right_hip'][0]) // 2,
                (body_landmarks['left_hip'][1] + body_landmarks['right_hip'][1]) // 2
            ])
        )
        size = self.estimate_size(shoulder_width)

        # Display measurements on frame
        cv2.putText(frame, f"Shoulder width: {int(shoulder_width)} px", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Torso height: {int(torso_height)} px", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Suggested size: {size}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return shoulder_width, torso_height, size

    def run(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        h, w = frame.shape[:2]

        measurements = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_landmarks = {
                'left_shoulder': (int(landmarks[11].x * w), int(landmarks[11].y * h)),
                'right_shoulder': (int(landmarks[12].x * w), int(landmarks[12].y * h)),
                'left_hip': (int(landmarks[23].x * w), int(landmarks[23].y * h)),
                'right_hip': (int(landmarks[24].x * w), int(landmarks[24].y * h)),
            }

            # Draw body landmarks and measurements
            shoulder_width, torso_height, size = self.draw_body_landmarks(frame, body_landmarks)
            measurements = {
                'shoulder_width': int(shoulder_width),
                'torso_height': int(torso_height),
                'size': size
            }

            # Warp and blend cloth
            warped_cloth = self.warp_cloth(body_landmarks, frame)
            frame = self.blend(frame, warped_cloth)

        return frame, measurements

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()

# ----------- COMMAND-LINE INTERFACE -----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Virtual Try-On with AR")
    parser.add_argument("--image", type=str, help="Path to the outfit image", required=False)
    args = parser.parse_args()
    
    # Use provided image path or default
    cloth_image_path = args.image if args.image and os.path.exists(args.image) else TRANSPARENT_IMAGE_PATH
    
    # If image provided, process it with background removal
    if args.image and os.path.exists(args.image):
        print(f"Processing outfit image: {args.image}")
        try:
            original_image = Image.open(args.image).convert("RGB")
            # We need to save the intermediate file to a valid path
            transparent_cloth = remove_background_and_save(original_image, TRANSPARENT_IMAGE_PATH)
            print("[OK] Background removed successfully")
            cloth_image_path = TRANSPARENT_IMAGE_PATH
        except Exception as e:
            print(f"[ERROR] Error processing image: {e}")
            print("Using original image without background removal")
            cloth_image_path = args.image
            
    if not os.path.exists(cloth_image_path):
        print(f"[ERROR] Error: Cloth image not found at {cloth_image_path}")
        print("Please provide a valid image path using --image argument")
        exit(1)
        
    print(f"Starting Virtual Try-On with: {cloth_image_path}")
    
    try:
        # Initialize application
        app = VirtualTryOn(cloth_path=cloth_image_path)
        
        # Start camera capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Error: Cannot access camera. Please check permissions.")
            exit(1)
            
        print("[OK] Camera started successfully")
        print("Press 'q' to quit, 's' to take screenshot")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Error: Cannot read from camera")
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            # Application Logic
            # 1. Detect body and measurements
            # 2. Warp cloth to body
            # 3. Display result
            try:
                result_frame, measurements = app.run(frame)
            except Exception as e:
                # If tracking fails, show original frame
                result_frame = frame
            
            # Simple placeholder logic until full implementation is restored
            cv2.imshow('Virtual Try-On - Press Q to quit', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_path = os.path.join(UPLOAD_FOLDER, f"screenshot_{screenshot_count}.png")
                cv2.imwrite(screenshot_path, frame)
                print(f"[OK] Screenshot saved: {screenshot_path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camera stopped.")
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)