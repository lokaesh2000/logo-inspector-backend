from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = FastAPI()

# --- CORS SETUP ---
# Allows your Vercel frontend to communicate with this Railway backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SETTINGS ---
# IMPORTANT: Update these to local relative paths for your GitHub/Railway repository!
model = YOLO('best.pt') 

# Map Class IDs to Master Images
masters = {
    0: cv2.imread('a1.png'),       # Amazon Master
    1: cv2.imread('fedex1.png'),   # FedEx Master
    2: cv2.imread('g0.png'),       # Google Master
    3: cv2.imread('n0.png')        # Nike Master
}

# --- HELPER FUNCTIONS ---
def rotate_image_bound(image, angle):
    """Rotates an image safely by an arbitrary angle without chopping the corners."""
    if angle == 0:
        return image.copy(), None

    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rot_img = cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))

    M_inv = cv2.getRotationMatrix2D((nW / 2, nH / 2), -angle, 1.0)
    M_inv[0, 2] += (w / 2) - (nW / 2)
    M_inv[1, 2] += (h / 2) - (nH / 2)

    return rot_img, M_inv

def extract_difference_mask(m_aligned, g_c):
    """Reusable function to generate the red pixel mask."""
    if m_aligned is None:
        return None
    g_m_aligned = cv2.cvtColor(m_aligned, cv2.COLOR_BGR2GRAY)
    g1_blur = cv2.GaussianBlur(g_m_aligned, (5, 5), 0)
    g2_blur = cv2.GaussianBlur(g_c, (5, 5), 0)

    diff = cv2.absdiff(g1_blur, g2_blur)
    _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def get_ink_bounds(gray_img):
    """Finds the true bounding box of the non-white ink, ignoring padding."""
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0

    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

    return x_min, y_min, x_max - x_min, y_max - y_min

def get_defects(crop, master):
    """Tournament Logic: Runs SIFT and Template Match, picks the one with fewer errors."""
    g_m = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    g_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h_c, w_c = g_c.shape

    # --- CANDIDATE 1: SIFT ALIGNMENT ---
    mask_sift = None
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g_m, None)
    kp2, des2 = sift.detectAndCompute(g_c, None)

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [match[0] for match in matches if len(match) == 2 and match[0].distance < 0.75 * match[1].distance]

        if len(good) > 3:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if M is not None:
                m_aligned_sift = cv2.warpAffine(master, M, (w_c, h_c), borderValue=(255,255,255))
                mask_sift = extract_difference_mask(m_aligned_sift, g_c)

    # --- CANDIDATE 2: TEMPLATE MATCHING ALIGNMENT (INK-ISOLATED) ---
    mask_template = None
    x_c, y_c, ink_w_c, ink_h_c = get_ink_bounds(g_c)
    x_m, y_m, ink_w_m, ink_h_m = get_ink_bounds(g_m)

    if ink_w_c > 0 and ink_w_m > 0:
        master_ink_crop = master[y_m:y_m+ink_h_m, x_m:x_m+ink_w_m]
        scale = min(ink_w_c / float(ink_w_m), ink_h_c / float(ink_h_m))
        new_w = max(1, int(ink_w_m * scale))
        new_h = max(1, int(ink_h_m * scale))
        
        if 0.1 < scale < 10.0:
            master_ink_scaled = cv2.resize(master_ink_crop, (new_w, new_h))
            g_master_ink_scaled = cv2.cvtColor(master_ink_scaled, cv2.COLOR_BGR2GRAY)
            
            if new_w <= w_c and new_h <= h_c:
                res = cv2.matchTemplate(g_c, g_master_ink_scaled, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                x_offset, y_offset = max_loc
                
                m_aligned_temp = np.full((h_c, w_c, 3), 255, dtype=np.uint8)
                paste_h = min(new_h, h_c - y_offset)
                paste_w = min(new_w, w_c - x_offset)
                m_aligned_temp[y_offset:y_offset+paste_h, x_offset:x_offset+paste_w] = master_ink_scaled[:paste_h, :paste_w]
                
                mask_template = extract_difference_mask(m_aligned_temp, g_c)

    # --- TOURNAMENT: WHO WON? ---
    sift_pixels = np.count_nonzero(mask_sift) if mask_sift is not None else float('inf')
    temp_pixels = np.count_nonzero(mask_template) if mask_template is not None else float('inf')

    if sift_pixels == float('inf') and temp_pixels == float('inf'):
        return None

    best_mask = mask_sift if sift_pixels < temp_pixels else mask_template

    # --- SAFETY VALVE ---
    if np.count_nonzero(best_mask) > 0.20 * best_mask.size:
        return None

    return best_mask

# --- API ENDPOINT ---
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image from frontend byte stream
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    full_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if full_img is None:
        return {"error": "Image could not be processed. Please try again."}

    final_output = full_img.copy()
    orig_h, orig_w = full_img.shape[:2]

    global_defect_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    total_logos_detected = 0

    # Execute 8-Angle Scanning
    for angle in angles:
        rot_img, M_inv = rotate_image_bound(full_img, angle)
        rot_defect_mask = np.zeros(rot_img.shape[:2], dtype=np.uint8)

        results = model.predict(rot_img, conf=0.4, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cid = int(box.cls[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rot_img.shape[1], x2), min(rot_img.shape[0], y2)

            logo_crop = rot_img[y1:y2, x1:x2]
            master_img = masters.get(cid)

            if master_img is not None and logo_crop.size > 0:
                mask = get_defects(logo_crop, master_img)

                if mask is not None:
                    total_logos_detected += 1
                    rot_defect_mask[y1:y2, x1:x2] = cv2.bitwise_or(rot_defect_mask[y1:y2, x1:x2], mask)

        if M_inv is not None:
            unrotated_mask = cv2.warpAffine(rot_defect_mask, M_inv, (orig_w, orig_h), flags=cv2.INTER_NEAREST)
        else:
            unrotated_mask = rot_defect_mask.copy()

        global_defect_mask = cv2.bitwise_or(global_defect_mask, unrotated_mask)

    if total_logos_detected == 0:
        return {"error": "No recognized logos detected. Please upload another image."}

    # Paint defects
    final_output[global_defect_mask > 0] = [0, 0, 255]

    contours, _ = cv2.findContours(global_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_data = [] 

    # Extract coordinates (no total count tracking)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(final_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            defect_data.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

    # Convert finalized image to Base64 for React frontend
    _, buffer = cv2.imencode('.jpg', final_output)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    base64_image = f"data:image/jpeg;base64,{b64_str}"

    # Return pure JSON structure (Image + Coordinates)
    return {
        "status": "success",
        "defect_coordinates": defect_data,
        "visual_proof": base64_image
    }