# 🔍 AI Logo Defect Inspector

An enterprise-grade Computer Vision pipeline designed to automatically detect, align, and inspect corporate logos (Amazon, FedEx, Google, Nike) for printing defects, scratches, and geometric precision. 

Built to handle extreme edge cases, this system processes images at any orientation, overcomes loose AI bounding boxes, and filters out sub-pixel mathematical noise to generate production-ready visual proofs and JSON coordinate data.

---

## 🧠 The Problem & How I Solved It

Standard object detection models (like YOLO) are great at finding logos, but they draw straight, axis-aligned boxes that are often "loose" (containing padding). When you try to compare a warped, tilted YOLO crop to a perfect Master image to find defects, simple alignment algorithms fail completely:
* **Template Matching** fails because YOLO's padding causes incorrect scaling.
* **SIFT (Feature Matching)** fails on repetitive text (e.g., matching the wrong 'e' in FedEx), causing massive distortion.
* **Rotated Logos** force straight bounding boxes to become giant squares, ruining aspect ratios.

### The Solution Architecture
I engineered a multi-stage pipeline to make the inspection bulletproof:

1. **8-Angle Rotational Scanning:** The system safely rotates the input image through 8 different orientations (0°, 45°, 90°, etc.). This ensures that no matter how the logo was photographed, it will eventually be processed perfectly upright.
2. **The "Ink Isolator":** Before scaling, the pipeline converts the YOLO crop to binary and mathematically shrinks the bounding box to the exact edges of the physical "ink." This guarantees perfect 1:1 scaling against the Master image, completely ignoring YOLO's sloppy padding.
3. **The Algorithmic "Tournament":** Instead of guessing which alignment method to use, the system runs both simultaneously. It generates a defect mask using **SIFT**, and a second mask using **Ink-Isolated Template Matching**. The algorithm then counts the error pixels and automatically declares the method with the fewest errors the "winner." 
4. **Anti-Aliasing Noise Filters:** Uses morphological operations and contour area thresholds (`>100px`) to ignore microscopic sub-pixel alignment errors, ensuring only true physical defects are flagged.

---

## 💻 Tech Stack

* **AI/ML:** Ultralytics (YOLOv8)
* **Computer Vision:** OpenCV (cv2), Numpy
* **Backend API (Production):** FastAPI, Uvicorn, Docker (Hosted on Railway)
* **Frontend UI (Production):** React, Vite (Hosted on Vercel)

---

## 🚀 How to Run Locally (CLI Version)

If you want to test the raw Computer Vision pipeline on your local machine, use the provided `test.py` script.

### 1. Clone the repository
```bash
git clone [https://github.com/lokaesh2000/logo-inspector-backend.git](https://github.com/lokaesh2000/logo-inspector-backend.git)
cd logo-inspector-backend
```

### 2. Set up a Python Virtual Environment (Recommended)
This prevents system-wide package conflicts (resolving PEP 668 errors on macOS).
```bash
python3 -m venv venv
source venv/bin/activate
```
