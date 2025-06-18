import cv2
import os
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
folder_path = r"C:\Users\hp\Desktop\crackpredection"  # <-- UPDATE THIS PATH
output_folder = os.path.join(folder_path, "output")
os.makedirs(output_folder, exist_ok=True)

fudgefactor = 1.3
sigma = 21
kernel = 2 * math.ceil(2 * sigma) + 1
font = cv2.FONT_HERSHEY_SIMPLEX

# === Initialize accumulators ===
all_true_labels = []
all_pred_labels = []

# === Process Each Image ===
for file in os.listdir(folder_path):
    if not file.lower().endswith(".jpg") or "_mask" in file:
        continue

    image_path = os.path.join(folder_path, file)
    print(f"\n🔍 Processing: {file}")

    # === Read Image ===
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("❌ Could not load image. Skipping.")
        continue

    # === Preprocessing (Edge Detection) ===
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)

    # === Analyze Edge Density for Uncracked Surface ===
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = (edge_pixels / total_pixels) * 100

    # === Basic Classification ===
    if edge_density < 0.5:
        surface_status = "✅ Surface is UNCRACKED."
        cracked = False
    else:
        surface_status = "⚠ Cracks detected on surface."
        cracked = True

    print(f"Edge Density: {edge_density:.4f}% --> {surface_status}")

    # === Advanced Crack Detection (Using Sobel + Morphology) ===
    gray_norm = gray / 255.0
    blur = cv2.GaussianBlur(gray_norm, (kernel, kernel), sigma)
    enhanced = cv2.subtract(gray_norm, blur)

    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)

    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0
    mag[mag >= threshold] = 255
    edge_mask = mag.astype(np.uint8)

    dilated = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
    crack_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))

    uncracked_mask = cv2.bitwise_not(crack_mask)

    # === Evaluation Metrics ===
    true_crack = (edge_mask > 127).astype(np.uint8).flatten()
    pred_crack = (crack_mask > 0).astype(np.uint8).flatten()

    # Accumulate labels for overall metrics
    all_true_labels.extend(true_crack)
    all_pred_labels.extend(pred_crack)

    # Compute per-image metrics
    if np.all(true_crack == 0) and np.all(pred_crack == 0):
        acc = prec = rec = f1 = 100.0
    else:
        acc = accuracy_score(true_crack, pred_crack) * 100
        prec = precision_score(true_crack, pred_crack, zero_division=0) * 100
        rec = recall_score(true_crack, pred_crack, zero_division=0) * 100
        f1 = f1_score(true_crack, pred_crack, zero_division=0) * 100

    # === Visualization ===
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = (0, 0, 255)
    yellow = (0, 255, 255)

    # Highlight cracks
    output_img[crack_mask > 0] = red

    # Add Transparent Metrics Box
    overlay = output_img.copy()
    cv2.rectangle(overlay, (10, 10), (370, 140), (0, 0, 255), -1)
    output_img = cv2.addWeighted(overlay, 0.3, output_img, 0.7, 0)

    # Write Metrics
    cv2.putText(output_img, f'Edge Density: {edge_density:.2f}%', (20, 30), font, 0.5, yellow, 1)
    cv2.putText(output_img, f'Accuracy: {acc:.2f}%', (20, 55), font, 0.5, yellow, 1)
    cv2.putText(output_img, f'Precision: {prec:.2f}%', (20, 80), font, 0.5, yellow, 1)
    cv2.putText(output_img, f'Recall: {rec:.2f}%', (20, 105), font, 0.5, yellow, 1)
    cv2.putText(output_img, f'F1 Score: {f1:.2f}%', (20, 130), font, 0.5, yellow, 1)

    # Write Final Message
    final_message = "Uncracked Surface" if not cracked else "Cracks Detected"
    color = (255, 0, 0) if not cracked else (0, 0, 255)
    cv2.putText(output_img, final_message, (20, output_img.shape[0] - 20), font, 0.6, color, 2)

    # === Save and Show Output ===
    save_path = os.path.join(output_folder, f"output_{file}")
    cv2.imwrite(save_path, output_img)

    cv2.imshow(f"Crack Detection - {file}", output_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === Terminal Output ===
    print("📊 Evaluation:")
    print(f"  ✔ Accuracy : {acc:.2f}%")
    print(f"  ✔ Precision: {prec:.2f}%")
    print(f"  ✔ Recall   : {rec:.2f}%")
    print(f"  ✔ F1 Score : {f1:.2f}%")
    print(f"  📝 Result: {final_message}")
    print(f"  💾 Saved to: {save_path}")

# === Overall Evaluation ===
print("\n🔍 Overall Evaluation Across All Images:")

# Compute overall confusion matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[1, 0])
TP, FN = cm[0]
FP, TN = cm[1]

# Compute overall metrics
overall_acc = accuracy_score(all_true_labels, all_pred_labels) * 100
overall_prec = precision_score(all_true_labels, all_pred_labels, zero_division=0) * 100
overall_rec = recall_score(all_true_labels, all_pred_labels, zero_division=0) * 100
overall_f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0) * 100

print("📊 Overall Metrics:")
print(f"  ✔ Accuracy : {overall_acc:.2f}%")
print(f"  ✔ Precision: {overall_prec:.2f}%")
print(f"  ✔ Recall   : {overall_rec:.2f}%")
print(f"  ✔ F1 Score : {overall_f1:.2f}%")

print("\nConfusion Matrix:")
print("Predicted")
print("Cracked  Uncracked")
print(f"Actual Cracked     {TP:5d}     {FN:5d}")
print(f"Actual Uncracked   {FP:5d}     {TN:5d}")

# === Plot and Save Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cracked', 'Uncracked'],
            yticklabels=['Cracked', 'Uncracked'])
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
conf_matrix_path = os.path.join(output_folder, 'overall_confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.show()
print(f"💾 Confusion matrix saved to: {conf_matrix_path}")
