import os
import cv2
import numpy as np

src_path = r"C:\Users\JAYANT\.gemini\antigravity-ide\brain\062f1838-7925-4857-a424-bec1fbb2f319\.user_uploaded\media_1788883952276.jpg"
out_dir = r"c:\Users\JAYANT\OneDrive\Desktop\onboard_bas_har_auto_prototype\gui\assets"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(src_path)
h, w = img.shape[:2]

# Detect orange chevron accurately
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_orange = np.array([5, 80, 80])
upper_orange = np.array([28, 255, 255])
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Dilate orange mask slightly so no orange pixels get tinted cyan
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
orange_dilated = cv2.dilate(orange_mask, kernel, iterations=1)

# Gray and lightness
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv_gray = 255.0 - gray.astype(np.float32)

# Alpha for text (excluding orange)
text_alpha = np.clip((inv_gray - 30.0) / (200.0 - 30.0), 0.0, 1.0)
text_alpha = (text_alpha ** 0.8) * 255.0
text_alpha[orange_dilated > 0] = 0

# Alpha for orange
# In orange region, distance from white
orange_alpha = np.clip((inv_gray - 20.0) / (160.0 - 20.0), 0.0, 1.0)
orange_alpha = (orange_alpha ** 0.8) * 255.0

# 1. Premium Spacecraft Theme: Diamond Ice-Cyan Gradient
# Letters fade from #FFFFFF (top) to #00F0FF (bottom)
out_spacecraft = np.zeros((h, w, 4), dtype=np.uint8)

# Vertical gradient factor from 0.0 (top) to 1.0 (bottom)
y_coords = np.linspace(0.0, 1.0, h)[:, np.newaxis]

# Interpolate between pure white (255, 255, 255) and neon cyan (255, 240, 0 in BGR)
grad_b = np.full((h, w), 255, dtype=np.float32)
grad_g = 255.0 - y_coords * 15.0 # 255 -> 240
grad_r = 255.0 - y_coords * 220.0 # 255 -> 35

# Apply to text
out_spacecraft[:, :, 0] = np.clip(grad_b, 0, 255).astype(np.uint8)
out_spacecraft[:, :, 1] = np.clip(grad_g, 0, 255).astype(np.uint8)
out_spacecraft[:, :, 2] = np.clip(grad_r, 0, 255).astype(np.uint8)
out_spacecraft[:, :, 3] = text_alpha.astype(np.uint8)

# Apply electric orange to chevron
orange_pixels = orange_mask > 30
# Electric aerospace orange: B=0, G=130, R=255
out_spacecraft[orange_pixels, 0] = 0
out_spacecraft[orange_pixels, 1] = 135
out_spacecraft[orange_pixels, 2] = 255
out_spacecraft[orange_pixels, 3] = np.clip(orange_alpha[orange_pixels] * 1.2, 0, 255).astype(np.uint8)

# Combined alpha mask for cropping
combined_alpha = np.maximum(out_spacecraft[:, :, 3], orange_mask)
coords = cv2.findNonZero(combined_alpha)
x, y, cw, ch = cv2.boundingRect(coords)
pad_x, pad_y = 20, 12
x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
x2, y2 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)

crop_spacecraft = out_spacecraft[y1:y2, x1:x2]

# Save cropped version
logo_path = os.path.join(out_dir, "tara_logo.png")
cv2.imwrite(logo_path, crop_spacecraft)

# Also save high-res base64 string for direct zero-latency embedding in HTML
import base64
_, buffer = cv2.imencode('.png', crop_spacecraft)
b64_str = base64.b64encode(buffer).decode('utf-8')
with open(os.path.join(out_dir, "tara_logo_b64.txt"), "w") as f:
    f.write(b64_str)

print("Saved tara_logo.png and base64. Cropped size:", crop_spacecraft.shape)
