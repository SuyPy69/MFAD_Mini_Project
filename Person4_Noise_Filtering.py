import numpy as np
import matplotlib.pyplot as plt
import cv2


# ============================================
# 1. LOAD THE ORIGINAL CHECKERS IMAGE
# ============================================

ImJPG = cv2.imread("checkers.pgm", cv2.IMREAD_GRAYSCALE)

if ImJPG is None:
    print("Error: checkers.pgm was not found.")
    exit()

print("Image loaded successfully!")
print("Image dimensions:", ImJPG.shape)


# ============================================
# 2. DISPLAY ORIGINAL IMAGE
# ============================================

plt.figure(figsize=(6, 6))
plt.imshow(ImJPG, cmap="gray")
plt.title("Original Checkers Image")
plt.axis("off")
plt.show()


# ============================================
# 3. GENERATE NOISE
# ============================================

m, n = ImJPG.shape

ImJPG_Noisy = (
    ImJPG.astype(np.float64)
    + 50 * (np.random.rand(m, n) - 0.5)
)

# Keep pixel values between 0 and 255
ImJPG_Noisy = np.clip(ImJPG_Noisy, 0, 255)


# ============================================
# 4. DISPLAY ORIGINAL AND NOISY IMAGE
# ============================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(ImJPG, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ImJPG_Noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================
# 5. COMPUTE SVD OF THE NOISY IMAGE
# ============================================

UIm, SIm, VIm = np.linalg.svd(
    ImJPG_Noisy,
    full_matrices=False
)

print("\nSVD completed!")

print("U dimensions:", UIm.shape)
print("Number of singular values:", len(SIm))
print("V dimensions:", VIm.shape)


# ============================================
# 6. PLOT SINGULAR VALUES
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(SIm)

plt.title("Singular Values of Noisy Image")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")

plt.grid(True)
plt.show()


# ============================================
# 7. FUNCTION FOR SVD DENOISING
# ============================================

def denoise_image(U, S, V, k):

    image = np.dot(
        U[:, :k],
        np.dot(
            np.diag(S[:k]),
            V[:k, :]
        )
    )

    return image


# ============================================
# 8. DENOISING USING k = 10, 30, 50
# ============================================

k_values = [10, 30, 50]

denoised_images = {}

for k in k_values:

    denoised = denoise_image(UIm, SIm, VIm, k)

    denoised_images[k] = denoised

    plt.figure(figsize=(6, 6))

    plt.imshow(denoised, cmap="gray")

    plt.title(f"Denoised Image - k = {k}")

    plt.axis("off")

    plt.show()


# ============================================
# 9. FINAL COMPARISON
# ============================================

plt.figure(figsize=(16, 4))

plt.subplot(1, 5, 1)
plt.imshow(ImJPG, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(ImJPG_Noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(denoised_images[10], cmap="gray")
plt.title("k = 10")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(denoised_images[30], cmap="gray")
plt.title("k = 30")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(denoised_images[50], cmap="gray")
plt.title("k = 50")
plt.axis("off")

plt.tight_layout()
plt.show()