import cv2
import numpy as np
import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Enable CORS for your React Frontend on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load YOLO26 Model
MODEL_PATH = "weights/best.pt"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Pre-load Digital Masters into memory
# Mapping Class IDs (0-4) to their master images
masters = {}
for i in range(5):
    path = f"masters/{i}.png"
    if os.path.exists(path):
        masters[i] = cv2.imread(path)
    else:
        print(f"Warning: Master image {path} missing.")

def run_inspection(crop, master):
    """Core SIFT alignment and defect detection logic."""
    gray_master = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_master, None)
    kp2, des2 = sift.detectAndCompute(gray_crop, None)

    # If no descriptors found, skip
    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Align Master to the Camera Crop
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            return None

        h, w = gray_crop.shape
        master_aligned = cv2.warpAffine(master, M, (w, h), borderValue=(255, 255, 255))
        
        # Difference Logic
        m_aligned_gray = cv2.cvtColor(master_aligned, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce camera noise
        m_blur = cv2.GaussianBlur(m_aligned_gray, (5, 5), 0)
        c_blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        
        diff = cv2.absdiff(m_blur, c_blur)
        _, thresh = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return mask
    return None

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    # Read the uploaded image
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    output_img = img.copy()

    # 3. YOLO26 Detection
    results = model.predict(img, conf=0.4)[0]

    # 4. Loop through each detected logo
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        # Get the Crop and its Master
        logo_crop = img[y1:y2, x1:x2]
        master_img = masters.get(cls_id)

        if master_img is not None and logo_crop.size > 0:
            defect_mask = run_inspection(logo_crop, master_img)
            
            if defect_mask is not None:
                # Apply Red Defects to the ROI in the output image
                roi = output_img[y1:y2, x1:x2]
                roi[defect_mask > 0] = [0, 0, 255] 
                
                # Draw the "Shrink-Wrapped" Green Bounding Box
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 5. Encode and return the resulting image
    _, buffer = cv2.imencode('.jpg', output_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)