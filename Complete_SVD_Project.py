import numpy as np
import matplotlib.pyplot as plt
import cv2

def basic_svd_demo():
    A = np.array([[2, 1], [-1, 1]], dtype=float)
    U, S, VT = np.linalg.svd(A)
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, S)
    reconstructed_A = U @ Sigma @ VT
    print("\n=== PART 1: BASIC SVD ===")
    print("A:\n", A)
    print("\nU:\n", U)
    print("\nSingular values:\n", S)
    print("\nSigma:\n", Sigma)
    print("\nV^T:\n", VT)
    print("\nU Sigma V^T:\n", reconstructed_A)
    print("\nA = U Sigma V^T:", np.allclose(A, reconstructed_A))
    print("\nU^T U:\n", U.T @ U)
    print("\nV^T V:\n", VT @ VT.T)
    print("U orthogonal:", np.allclose(U.T @ U, np.eye(U.shape[1])))
    print("V orthogonal:", np.allclose(VT.T @ VT, np.eye(VT.shape[0])))
    return A

def geometric_visualization(A):
    U, S, VT = np.linalg.svd(A)
    Sigma = np.diag(S)
    theta = np.linspace(0, 2*np.pi, 300)
    circle = np.array([np.cos(theta), np.sin(theta)])
    after_VT = VT @ circle
    after_Sigma = Sigma @ after_VT
    after_U = U @ after_Sigma
    direct_A = A @ circle
    print("\n=== PART 2: GEOMETRIC INTERPRETATION ===")
    print("Verification:", np.allclose(A, U @ Sigma @ VT))
    points = np.concatenate([circle, after_VT, after_Sigma, after_U, direct_A], axis=1)
    lim = np.max(np.abs(points)) + 0.5
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plots = [
        (circle, "Original Unit Circle"),
        (after_VT, r"After $V^T$"),
        (after_Sigma, r"After $\Sigma$"),
        (after_U, "After U"),
        (direct_A, "Direct Transformation by A")
    ]
    for ax, (p, title) in zip(axes, plots):
        ax.plot(p[0], p[1])
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(True)
        ax.axhline(0, linewidth=.8); ax.axvline(0, linewidth=.8)
    plt.suptitle("Geometric Interpretation of SVD")
    plt.tight_layout()
    plt.show()

def image_compression(image_file="einstein.jpg"):
    ImJPG = plt.imread(image_file)
    if ImJPG.ndim == 3:
        ImJPG = ImJPG[:, :, :3]
        ImJPG = (0.299*ImJPG[:,:,0] + 0.587*ImJPG[:,:,1] + 0.114*ImJPG[:,:,2])
    ImJPG = ImJPG.astype(float)
    if ImJPG.max() <= 1: ImJPG *= 255
    m, n = ImJPG.shape
    print("\n=== PART 3: IMAGE COMPRESSION ===")
    print("Image matrix size:", m, "x", n)
    plt.figure(figsize=(6,6)); plt.imshow(ImJPG, cmap="gray"); plt.title("Original Einstein Image"); plt.axis("off"); plt.show()
    UIm, SIm, VIm = np.linalg.svd(ImJPG, full_matrices=False)
    plt.figure(figsize=(8,5)); plt.plot(np.arange(1,len(SIm)+1), SIm); plt.title("Singular Values of Einstein Image"); plt.xlabel("Singular value number"); plt.ylabel("Singular value"); plt.grid(); plt.show()
    for k in [50,100,150]:
        compressed = UIm[:,:k] @ np.diag(SIm[:k]) @ VIm[:k,:]
        compressed = np.clip(compressed,0,255)
        plt.figure(figsize=(6,6)); plt.imshow(compressed,cmap="gray"); plt.title(f"Compressed Image - k = {k}"); plt.axis("off"); plt.show()
        original_values = m*n
        compressed_values = m*k + k*n
        percentage = (1-compressed_values/original_values)*100
        print(f"k={k} | original={original_values} | compressed={compressed_values} | compression={percentage:.2f}%")

def denoise_image(U,S,V,k):
    return U[:,:k] @ np.diag(S[:k]) @ V[:k,:]

def noise_filtering(image_file="checkers.pgm"):
    ImJPG = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if ImJPG is None:
        print("\nError: checkers.pgm was not found.")
        return
    print("\n=== PART 4: NOISE FILTERING ===")
    print("Image loaded successfully!")
    print("Image dimensions:", ImJPG.shape)
    plt.figure(figsize=(6,6)); plt.imshow(ImJPG,cmap="gray"); plt.title("Original Checkers Image"); plt.axis("off"); plt.show()
    m,n = ImJPG.shape
    noisy = ImJPG.astype(float) + 50*(np.random.rand(m,n)-0.5)
    noisy = np.clip(noisy,0,255)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.imshow(ImJPG,cmap="gray"); plt.title("Original Image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(noisy,cmap="gray"); plt.title("Noisy Image"); plt.axis("off")
    plt.tight_layout(); plt.show()
    U,S,V = np.linalg.svd(noisy, full_matrices=False)
    plt.figure(figsize=(8,5)); plt.plot(S); plt.title("Singular Values of Noisy Image"); plt.xlabel("Singular Value Index"); plt.ylabel("Singular Value"); plt.grid(); plt.show()
    results={}
    for k in [10,30,50]:
        results[k] = np.clip(denoise_image(U,S,V,k),0,255)
        plt.figure(figsize=(6,6)); plt.imshow(results[k],cmap="gray"); plt.title(f"Denoised Image - k = {k}"); plt.axis("off"); plt.show()
    plt.figure(figsize=(16,4))
    for i,(title,img) in enumerate([("Original",ImJPG),("Noisy",noisy),("k = 10",results[10]),("k = 30",results[30]),("k = 50",results[50])],1):
        plt.subplot(1,5,i); plt.imshow(img,cmap="gray"); plt.title(title); plt.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    A = basic_svd_demo()
    geometric_visualization(A)
    image_compression("einstein.jpg")
    noise_filtering("checkers.pgm")
    print("\n=== COMPLETE SVD PROJECT FINISHED ===")
