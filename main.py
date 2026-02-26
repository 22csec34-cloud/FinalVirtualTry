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
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ── ADJUSTABLE FIT PARAMETERS ──
        self.top_width_pad = 20     # extra pixels added to each side of top
        self.top_height_pad = 10    # extra pixels added to bottom of top
        self.bottom_width_pad = 15  # extra pixels added to each side of bottom
        self.bottom_height_pad = 0  # extra pixels added to bottom of pants
        
        # Load cloth
        self.cloth = cv2.imread(self.cloth_path, cv2.IMREAD_UNCHANGED)
        if self.cloth is None:
            raise ValueError(f"Cannot load cloth image from {self.cloth_path}")
        
        # Ensure 4-channel (BGRA)
        if self.cloth.shape[2] == 3:
            self.cloth = cv2.cvtColor(self.cloth, cv2.COLOR_BGR2BGRA)
        
        # Segment the outfit into top and bottom
        self.top_img, self.bottom_img = self._segment_outfit(self.cloth)
        
        if self.top_img is not None:
            print(f"[INFO] Top garment: {self.top_img.shape[1]}x{self.top_img.shape[0]}", flush=True)
        if self.bottom_img is not None:
            print(f"[INFO] Bottom garment: {self.bottom_img.shape[1]}x{self.bottom_img.shape[0]}", flush=True)
        if self.top_img is None and self.bottom_img is None:
            raise ValueError("Could not segment any garment from the outfit image")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SEGMENTATION: Split outfit into top & bottom
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _segment_outfit(self, cloth):
        """
        Segment outfit into top and bottom garments.
        Strategy:
          1. Find contours in the alpha channel
          2. If 2+ large contours: top-most = top, bottom-most = bottom
          3. If 1 contour: scan for horizontal gap to split
          4. If no gap: treat as top only
        Returns (top_img, bottom_img) — either can be None.
        """
        alpha = cloth[:, :, 3].copy()
        img_h, img_w = alpha.shape
        
        # Clean up alpha
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cloth, None
        
        # Filter noise: keep contours > 3% of image area
        min_area = img_h * img_w * 0.03
        large = [c for c in contours if cv2.contourArea(c) > min_area]
        if not large:
            large = [max(contours, key=cv2.contourArea)]
        
        # Sort by vertical center
        bboxes = []
        for c in large:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w, h, y + h // 2))
        bboxes.sort(key=lambda b: b[4])
        
        if len(bboxes) >= 2:
            # Two separate garments detected
            top_box = bboxes[0]
            bot_box = bboxes[-1]
            top_img = self._crop_piece(cloth, top_box[:4])
            bot_img = self._crop_piece(cloth, bot_box[:4])
            print("[INFO] Detected 2 separate garment pieces (top + bottom)", flush=True)
            return top_img, bot_img
        
        # Single contour — try gap-based split
        x, y, w, h, _ = bboxes[0]
        region_alpha = alpha[y:y+h, x:x+w]
        
        # Compute row-wise pixel density
        row_density = np.array([np.count_nonzero(region_alpha[r, :]) for r in range(h)])
        
        # Search for gap in the middle 50% of height
        s = int(h * 0.25)
        e = int(h * 0.75)
        gap_threshold = w * 0.05  # < 5% of width = empty
        
        gap_rows = [r for r in range(s, e) if row_density[r] < gap_threshold]
        
        if len(gap_rows) > 3:
            # Split at the middle of the gap
            split = gap_rows[len(gap_rows) // 2]
            top_img = self._crop_piece(cloth, (x, y, w, split))
            bot_img = self._crop_piece(cloth, (x, y + split, w, h - split))
            print(f"[INFO] Split single contour at row {y + split} into top + bottom", flush=True)
            return top_img, bot_img
        
        # No gap found — single garment (top only)
        top_img = self._crop_piece(cloth, (x, y, w, h))
        print("[INFO] Single garment detected — treating as top only", flush=True)
        return top_img, None

    def _crop_piece(self, cloth, bbox):
        """Crop a garment piece with small padding."""
        x, y, w, h = bbox
        pad = 3
        y1 = max(0, y - pad)
        y2 = min(cloth.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(cloth.shape[1], x + w + pad)
        piece = cloth[y1:y2, x1:x2].copy()
        if piece.size == 0:
            return None
        return piece

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  WARPING: Map garment corners directly to body points
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _overlay_piece(self, piece, dst_4pts, frame):
        """
        Warp a garment piece by mapping its CONTENT BOUNDING BOX to 4 body destination points.
        Uses alpha channel to find the actual garment pixels, not the full image corners.
        dst_4pts: [top-left, top-right, bottom-left, bottom-right] in frame coords.
        """
        if piece is None:
            return None
        
        ph, pw = piece.shape[:2]
        
        # Find the actual garment content bounds from alpha channel
        if piece.shape[2] == 4:
            alpha = piece[:, :, 3]
        else:
            alpha = cv2.cvtColor(piece[:, :, :3], cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get bounding box of the actual garment content
        largest = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest)
        
        # Use the CONTENT bounding box corners as source points
        # This ensures the garment stretches to exactly fill the body region
        src = np.float32([
            [cx, cy],              # top-left of content
            [cx + cw, cy],         # top-right of content
            [cx, cy + ch],         # bottom-left of content
            [cx + cw, cy + ch],    # bottom-right of content
        ])
        
        dst = np.float32(dst_4pts)
        
        M = cv2.getPerspectiveTransform(src, dst)
        out_h, out_w = frame.shape[:2]
        warped = cv2.warpPerspective(piece, M, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        return warped

    def warp_cloth(self, body_landmarks, frame):
        """
        Overlay the top and bottom garments onto the body.
        Top: mapped to shoulders → hips  
        Bottom: mapped to hips → ankles (or estimated if not visible)
        """
        out_h, out_w = frame.shape[:2]
        result = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        
        ls = body_landmarks['left_shoulder']
        rs = body_landmarks['right_shoulder']
        lh = body_landmarks['left_hip']
        rh = body_landmarks['right_hip']
        
        wp_top = self.top_width_pad
        hp_top = self.top_height_pad
        wp_bot = self.bottom_width_pad
        hp_bot = self.bottom_height_pad
        
        # ── TOP: corners → [left_shoulder, right_shoulder, left_hip, right_hip] ──
        if self.top_img is not None:
            top_dst = [
                [ls[0] - wp_top, ls[1] - 5],           # top-left: left shoulder (slight up offset)
                [rs[0] + wp_top, rs[1] - 5],           # top-right: right shoulder
                [lh[0] - wp_top, lh[1] + hp_top],     # bottom-left: left hip
                [rh[0] + wp_top, rh[1] + hp_top],     # bottom-right: right hip
            ]
            warped_top = self._overlay_piece(self.top_img, top_dst, frame)
            if warped_top is not None:
                mask = warped_top[:, :, 3] > 0
                result[mask] = warped_top[mask]
        
        # ── BOTTOM: corners → [left_hip, right_hip, left_ankle, right_ankle] ──
        if self.bottom_img is not None:
            lk = body_landmarks.get('left_knee')
            rk = body_landmarks.get('right_knee')
            la = body_landmarks.get('left_ankle')
            ra = body_landmarks.get('right_ankle')
            
            # Determine bottom anchor: prefer ankle, fallback to knee, then estimate
            if la and ra and la[1] > lh[1] and ra[1] > rh[1]:
                # Use ankles for full-length pants
                bot_y_left = la[1] + hp_bot
                bot_y_right = ra[1] + hp_bot
                bot_x_left = la[0] - wp_bot
                bot_x_right = ra[0] + wp_bot
            elif lk and rk and lk[1] > lh[1] and rk[1] > rh[1]:
                # Use knees — estimate ankle position below
                torso_h = ((lh[1] + rh[1]) // 2) - ((ls[1] + rs[1]) // 2)
                extend = int(torso_h * 0.7)  # extend below knee
                bot_y_left = lk[1] + extend + hp_bot
                bot_y_right = rk[1] + extend + hp_bot
                bot_x_left = lk[0] - wp_bot
                bot_x_right = rk[0] + wp_bot
            else:
                # Full fallback: estimate from torso
                torso_h = ((lh[1] + rh[1]) // 2) - ((ls[1] + rs[1]) // 2)
                leg_len = int(torso_h * 1.5)
                bot_y_left = lh[1] + leg_len + hp_bot
                bot_y_right = rh[1] + leg_len + hp_bot
                bot_x_left = lh[0] - wp_bot
                bot_x_right = rh[0] + wp_bot
            
            # Ensure bottom points are within frame
            bot_y_left = min(bot_y_left, out_h - 5)
            bot_y_right = min(bot_y_right, out_h - 5)
            
            bottom_dst = [
                [lh[0] - wp_bot, lh[1]],               # top-left: left hip
                [rh[0] + wp_bot, rh[1]],               # top-right: right hip
                [bot_x_left, bot_y_left],               # bottom-left: left ankle/estimate
                [bot_x_right, bot_y_right],             # bottom-right: right ankle/estimate
            ]
            warped_bot = self._overlay_piece(self.bottom_img, bottom_dst, frame)
            if warped_bot is not None:
                mask = warped_bot[:, :, 3] > 0
                # Don't overwrite where top already placed
                no_top = ~(result[:, :, 3] > 0)
                result[mask & no_top] = warped_bot[mask & no_top]
        
        return result

    def blend(self, frame, warped_cloth):
        if warped_cloth.shape[2] == 4:
            alpha = warped_cloth[:, :, 3] / 255.0
            alpha = np.stack([alpha, alpha, alpha], axis=2)
            cloth_rgb = warped_cloth[:, :, :3]
            blended = (alpha * cloth_rgb + (1 - alpha) * frame).astype(np.uint8)
            return blended
        return frame

    def estimate_size(self, shoulder_width):
        """Estimate T-shirt size based on shoulder width in pixels."""
        if shoulder_width < 100: return "S"
        elif shoulder_width < 130: return "M"
        elif shoulder_width < 160: return "L"
        elif shoulder_width < 190: return "XL"
        elif shoulder_width < 220: return "XXL"
        else: return "XXXL"

    def draw_body_landmarks(self, frame, body_landmarks):
        # Draw key points
        for name, pt in body_landmarks.items():
            color = (0, 0, 255) if 'shoulder' in name or 'hip' in name else (255, 150, 0)
            cv2.circle(frame, pt, 4, color, -1)

        # Upper body skeleton (green)
        cv2.line(frame, body_landmarks['left_shoulder'], body_landmarks['right_shoulder'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['left_hip'], body_landmarks['right_hip'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['left_shoulder'], body_landmarks['left_hip'], (144, 238, 144), 2)
        cv2.line(frame, body_landmarks['right_shoulder'], body_landmarks['right_hip'], (144, 238, 144), 2)
        
        # Lower body skeleton (orange)
        if 'left_knee' in body_landmarks and 'right_knee' in body_landmarks:
            cv2.line(frame, body_landmarks['left_hip'], body_landmarks['left_knee'], (255, 200, 100), 2)
            cv2.line(frame, body_landmarks['right_hip'], body_landmarks['right_knee'], (255, 200, 100), 2)
        if 'left_ankle' in body_landmarks and 'right_ankle' in body_landmarks:
            lk = body_landmarks.get('left_knee', body_landmarks['left_hip'])
            rk = body_landmarks.get('right_knee', body_landmarks['right_hip'])
            cv2.line(frame, lk, body_landmarks['left_ankle'], (200, 150, 50), 2)
            cv2.line(frame, rk, body_landmarks['right_ankle'], (200, 150, 50), 2)

        # Measurements
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

        # HUD
        cv2.putText(frame, f"Shoulder: {int(shoulder_width)}px  Torso: {int(torso_height)}px", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {size}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Top pad W:{self.top_width_pad} H:{self.top_height_pad}  "
                    f"Bot pad W:{self.bottom_width_pad} H:{self.bottom_height_pad}", 
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
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
                'left_knee': (int(landmarks[25].x * w), int(landmarks[25].y * h)),
                'right_knee': (int(landmarks[26].x * w), int(landmarks[26].y * h)),
                'left_ankle': (int(landmarks[27].x * w), int(landmarks[27].y * h)),
                'right_ankle': (int(landmarks[28].x * w), int(landmarks[28].y * h)),
            }

            # Draw body
            shoulder_width, torso_height, size = self.draw_body_landmarks(frame, body_landmarks)
            measurements = {
                'shoulder_width': int(shoulder_width),
                'torso_height': int(torso_height),
                'size': size
            }

            # Warp and blend
            warped = self.warp_cloth(body_landmarks, frame)
            frame = self.blend(frame, warped)

        return frame, measurements

    def adjust_fit(self, key):
        """
        Adjust padding with keyboard.
        Top:    A/D = width pad -/+    W/S = height pad +/-
        Bottom: LEFT/RIGHT = width pad -/+    UP/DOWN = height pad +/-
        R = reset
        """
        step = 5
        
        if key == ord('a') or key == ord('A'):
            self.top_width_pad = max(0, self.top_width_pad - step)
            return True
        elif key == ord('d') or key == ord('D'):
            self.top_width_pad = min(80, self.top_width_pad + step)
            return True
        elif key == ord('w') or key == ord('W'):
            self.top_height_pad = min(80, self.top_height_pad + step)
            return True
        elif key == ord('s') or key == ord('S'):
            self.top_height_pad = max(-30, self.top_height_pad - step)
            return True
        elif key == 81 or key == 2424832:  # LEFT
            self.bottom_width_pad = max(0, self.bottom_width_pad - step)
            return True
        elif key == 83 or key == 2555904:  # RIGHT
            self.bottom_width_pad = min(80, self.bottom_width_pad + step)
            return True
        elif key == 82 or key == 2490368:  # UP
            self.bottom_height_pad = min(80, self.bottom_height_pad + step)
            return True
        elif key == 84 or key == 2621440:  # DOWN
            self.bottom_height_pad = max(-30, self.bottom_height_pad - step)
            return True
        elif key == ord('r') or key == ord('R'):
            self.top_width_pad = 20
            self.top_height_pad = 10
            self.bottom_width_pad = 15
            self.bottom_height_pad = 0
            print("[INFO] Padding reset to defaults", flush=True)
            return True
        return False

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()

# ----------- COMMAND-LINE INTERFACE -----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Virtual Try-On with AR")
    parser.add_argument("--image", type=str, help="Path to the outfit image", required=False)
    args = parser.parse_args()
    
    cloth_image_path = args.image if args.image and os.path.exists(args.image) else TRANSPARENT_IMAGE_PATH
    
    if args.image and os.path.exists(args.image):
        print(f"Processing outfit image: {args.image}")
        try:
            original_image = Image.open(args.image).convert("RGB")
            transparent_cloth = remove_background_and_save(original_image, TRANSPARENT_IMAGE_PATH)
            print("[OK] Background removed successfully")
            cloth_image_path = TRANSPARENT_IMAGE_PATH
        except Exception as e:
            print(f"[ERROR] Error processing image: {e}")
            print("Using original image without background removal")
            cloth_image_path = args.image
            
    if not os.path.exists(cloth_image_path):
        print(f"[ERROR] Cloth image not found at {cloth_image_path}")
        print("Please provide a valid image path using --image argument")
        exit(1)
        
    print(f"Starting Virtual Try-On with: {cloth_image_path}")
    
    try:
        app = VirtualTryOn(cloth_path=cloth_image_path)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Cannot access camera.")
            exit(1)
            
        print("[OK] Camera started successfully")
        print("Controls:")
        print("  Top fit:    A/D = width pad    W/S = height pad")
        print("  Bottom fit: LEFT/RIGHT = width    UP/DOWN = height")
        print("  R = reset    Q = quit    P = screenshot")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read from camera")
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            try:
                result_frame, measurements = app.run(frame)
            except Exception as e:
                result_frame = frame
            
            cv2.putText(result_frame, "Top: WASD | Bottom: Arrows | R: Reset | Q: Quit",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow('Virtual Try-On', result_frame)
            
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
            elif key == ord('p') or key == ord('P'):
                screenshot_path = os.path.join(UPLOAD_FOLDER, f"screenshot_{screenshot_count}.png")
                cv2.imwrite(screenshot_path, result_frame)
                print(f"[OK] Screenshot saved: {screenshot_path}")
                screenshot_count += 1
            else:
                app.adjust_fit(key)
        
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camera stopped.")
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)