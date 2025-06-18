import os
# === Configuration ===
folder_path = r"C:\Users\hp\Desktop\crackpredection"  # <-- UPDATE THIS PATH
output_folder = os.path.join(folder_path, "output")
os.makedirs(output_folder, exist_ok=True)

# Processing Parameters
fudgefactor = 1.3
sigma = 21
kernel = 2 * round(2 * sigma) + 1  # Kernel size for Gaussian Blur
font = 0.6  # Font size for text in images
