import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# PERSON 3: IMAGE COMPRESSION USING SVD
# --------------------------------------------------

# 1. Load Einstein image
image_file = "einstein.jpg"

ImJPG = plt.imread(image_file)

# Convert RGB/RGBA image to grayscale if necessary
if ImJPG.ndim == 3:
    ImJPG = ImJPG[:, :, :3]
    ImJPG = (
        0.299 * ImJPG[:, :, 0]
        + 0.587 * ImJPG[:, :, 1]
        + 0.114 * ImJPG[:, :, 2]
    )

ImJPG = ImJPG.astype(np.float64)

# If image values are 0-1, convert them to 0-255
if ImJPG.max() <= 1:
    ImJPG = ImJPG * 255

m, n = ImJPG.shape

# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(ImJPG, cmap="gray")
plt.title("Original Einstein Image")
plt.axis("off")
plt.show()

# 2. Compute SVD
UIm, SIm, VIm = np.linalg.svd(ImJPG, full_matrices=False)

print("Image matrix size:", m, "x", n)
print("Number of singular values:", len(SIm))

# 3. Plot singular values
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(SIm) + 1), SIm)
plt.title("Singular Values of Einstein Image")
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
plt.grid()
plt.show()

# 4. Image compression using k = 50, 100, 150
for k in [50, 100, 150]:

    # Truncated SVD:
    # A_k = U_k Sigma_k V_k^T
    ImJPG_comp = (
        UIm[:, :k]
        @ np.diag(SIm[:k])
        @ VIm[:k, :]
    )

    # Keep pixel values valid
    ImJPG_comp = np.clip(ImJPG_comp, 0, 255)

    # Display compressed image
    plt.figure(figsize=(6, 6))
    plt.imshow(ImJPG_comp, cmap="gray")
    plt.title(f"Compressed Image - k = {k}")
    plt.axis("off")
    plt.show()

    # Compression calculation
    original_values = m * n
    compressed_values = m * k + k * n
    compression_fraction = 1 - compressed_values / original_values
    compression_percentage = compression_fraction * 100

    print("----------------------------------------")
    print("k =", k)
    print("Original values stored:", original_values)
    print("Compressed values stored:", compressed_values)
    print("Compression percentage:", round(compression_percentage, 2), "%")
