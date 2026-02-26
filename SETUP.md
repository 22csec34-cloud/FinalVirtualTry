# 🪞 Virtual Try-On Fashion — Setup & Run Guide

## Overview
This module provides **real-time AR virtual try-on** using your webcam. It consists of:
- **Python Backend** (`main.py`) — MediaPipe pose detection + cloth warping + background removal
- **React Frontend** (`frontend/`) — Web UI for uploading outfit images and launching AR

---

## Prerequisites

| Tool     | Version | Install Link                              |
|---------|---------|-------------------------------------------|
| Python  | 3.9+    | https://www.python.org/downloads/         |
| Node.js | v18+    | https://nodejs.org/                       |
| Webcam  | —       | Required for AR try-on                    |
| Git     | Latest  | https://git-scm.com/downloads             |

---

## Step-by-Step Setup

### Part A: Python Backend

#### 1. Create a Virtual Environment (Recommended)

```bash
# From the project root (FINAL_ONE/)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 2. Install Python Dependencies

```bash
cd virtual_try_fashion
pip install -r requirements.txt
```

#### 3. Run the Python AR Try-On

**Option 1: With a specific outfit image**
```bash
python main.py --image path/to/your/outfit.png
```

**Option 2: Using the default uploaded image**
```bash
python main.py
```
> This uses the default image at `uploads/image1.png`. Make sure an outfit image exists there.

#### 4. Controls (During AR Session)

| Key | Action              |
|-----|---------------------|
| `Q` | Quit the AR session |
| `S` | Take a screenshot   |

Screenshots are saved in the `uploads/` folder.

---

### Part B: React Frontend

#### 1. Install Dependencies

```bash
cd virtual_try_fashion/frontend
npm install
```

#### 2. Run the Frontend Dev Server

```bash
npm run dev
```

The frontend will start on **http://localhost:5173** (Vite default)

#### 3. Build for Production (Optional)

```bash
npm run build
```

---

## How It Works

```
┌──────────────────────────────────────────────────┐
│  1. Upload outfit image                          │
│  2. Background removed (rembg + u2net model)     │
│  3. Webcam captures your body                    │
│  4. MediaPipe detects body pose landmarks        │
│  5. Outfit warped to match your body proportions │
│  6. Real-time AR overlay on webcam feed          │
│  7. Size estimation based on shoulder width      │
└──────────────────────────────────────────────────┘
```

### Features
- **Background Removal** — Automatically removes outfit image background using rembg
- **Pose Detection** — MediaPipe detects shoulder and hip landmarks
- **Cloth Warping** — Perspective transform maps clothing to body
- **Size Estimation** — Suggests clothing size (S/M/L/XL/XXL/XXXL) based on shoulder width
- **Body Measurements** — Displays shoulder width and torso height in pixels
- **Screenshots** — Save AR try-on snapshots

---

## Tech Stack

| Technology     | Purpose                                     |
|---------------|---------------------------------------------|
| Python 3.9+   | Core backend language                       |
| OpenCV         | Video capture, image processing, AR display |
| MediaPipe      | Body pose landmark detection                |
| rembg          | AI-powered background removal               |
| Pillow         | Image format handling                        |
| NumPy          | Numerical computations                       |
| React          | Frontend UI framework                       |
| Vite           | Frontend build tool                          |
| react-dropzone | Drag-and-drop image upload                  |

---

## Project Structure

```
virtual_try_fashion/
├── main.py                   # Main Python AR try-on application
├── requirements.txt          # Python dependencies
├── requirement.txt           # Legacy file (use requirements.txt instead)
├── uploads/                  # Uploaded & processed images
│   └── image1.png            # Default processed outfit image
├── assets/                   # Static assets
├── frontend/                 # React frontend
│   ├── src/                  # React source code
│   ├── public/               # Static frontend assets
│   ├── package.json          # Node.js dependencies
│   ├── vite.config.js        # Vite configuration
│   └── index.html            # HTML entry point
└── .gitignore
```

---

## Integration with Backend

The virtual try-on can be triggered from the main **Style Weaver** app:
1. User generates an outfit in Style Weaver
2. Clicks "Virtual Try On" in the gallery
3. Backend calls `python main.py --image <generated_image_path>`
4. AR webcam session opens with the outfit

---

## Troubleshooting

| Issue                                    | Solution                                              |
|------------------------------------------|-------------------------------------------------------|
| `Cannot access camera`                   | Check webcam permissions & ensure no other app uses it |
| `Cannot load cloth image`                | Ensure outfit image exists at the specified path       |
| `ModuleNotFoundError: cv2`               | Run `pip install opencv-python`                        |
| `ModuleNotFoundError: mediapipe`         | Run `pip install mediapipe`                            |
| `rembg model download fails`             | Check internet connection, run `python download_rembg_model.py` from backend/ |
| `No contours found in cloth image`       | Image may be fully transparent; try a different outfit |
| `npm install` fails (frontend)           | Delete `node_modules/` and `package-lock.json`, retry  |
