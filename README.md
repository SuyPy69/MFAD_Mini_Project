# Singular Value Decomposition (SVD): Image Compression and Noise Filtering

## Project Overview

This project demonstrates **Singular Value Decomposition (SVD)**, a fundamental concept in linear algebra, and explores its applications in image processing. It covers the mathematical foundation of SVD, its geometric interpretation, image compression, and noise filtering using Python.

SVD decomposes a matrix into three matrices, allowing image data to be approximated using fewer components.

## Objectives

- Understand the mathematical foundation of SVD.
- Verify matrix decomposition and orthogonality properties using Python.
- Visualize the geometric interpretation of SVD.
- Demonstrate image compression using truncated SVD.
- Explore noise filtering by reconstructing images using selected singular values.

## Mathematical Foundation

Singular Value Decomposition represents a matrix \(A\) as:

\[
$A_{k}$ = $U_{k}$ $Σ_{k}$ $V_{k}$ᵀ
\]

Where:

- **A:** Original matrix.
- **U:** Matrix of left singular vectors.
- **Σ (Sigma):** Diagonal matrix containing singular values in descending order.
- **Vᵀ:** Transpose of the matrix of right singular vectors.

The matrices \(U\) and \(V\) are orthogonal, while the singular values in \(Σ) determine the scaling along corresponding directions.

## Project Modules

### 1. Basic SVD Demonstration

- Defines a sample matrix.
- Computes its SVD using NumPy.
- Displays the matrices \(U\), \(Σ), and \(Vᵀ).
- Reconstructs the original matrix.
- Verifies the reconstruction and orthogonality properties.

### 2. Geometric Interpretation

- Visualizes how a matrix transforms a unit circle.
- Demonstrates rotation, reflection, and scaling.
- Shows how a circle can transform into an ellipse.
- Explains how singular values determine the lengths of the ellipse's principal axes.

### 3. Image Compression

- Loads an Einstein image from `einstein.jpg`.
- Displays the singular values of the image matrix.
- Reconstructs the image using different numbers of singular values, such as 50, 100, and 150.
- Compares image quality at different levels of compression.

Truncated SVD approximates an image matrix as:

\[
$A_{k}$ = $U_{k}$ $Σ_{k}$ $V_{k}$ᵀ
\]

Keeping fewer singular values generally reduces the amount of information stored in the approximation, but may also reduce image quality.

### 4. Noise Filtering

- Loads a checkerboard image from `checkers.pgm`.
- Displays the original and noisy images.
- Examines the singular values of the image.
- Reconstructs the image using selected numbers of singular values, such as 10, 30, and 50.
- Compares the reconstructed images to demonstrate noise reduction.

Truncated SVD can reduce some noise by discarding smaller singular-value components. The effectiveness depends on the image and the type of noise.

## Technologies Used

- **Python** — Programming language
- **NumPy** — Matrix operations and SVD computation
- **Matplotlib** — Plotting and image visualization
- **OpenCV** — Image processing and loading

## Installation

Install the required Python libraries using pip:

```bash
pip install numpy matplotlib opencv-python
```

## Project Structure

Ensure the required input images are placed in the same directory as the Python script.

```text
PES_MFAD/
├── complete_SVD_project.py
├── einstein.jpg
└── checkers.pgm
```

- `complete_SVD_project.py` — Main Python program.
- `einstein.jpg` — Input image for compression.
- `checkers.pgm` — Input image for noise filtering.

**Note:** The image files must be available separately if they are not included with the project. Filenames and extensions must match those used in the code.

## How to Run

1. Install Python and the required libraries.
2. Open the project folder in PyCharm or another Python IDE.
3. Place `einstein.jpg` and `checkers.pgm` in the project directory.
4. Open `complete_SVD_project.py`.
5. Run the script.
6. Observe the printed matrix results and generated plots.

**Note:** Matplotlib may display figures one at a time. Close each figure window to allow execution to continue to the next visualization.

## Expected Results

- Successful decomposition and reconstruction of a sample matrix.
- Verification of orthogonality properties.
- Geometric visualization of SVD transformations.
- Image compression comparisons using different values of \(k\).
- Comparisons of original, noisy, and reconstructed images.

The results depend on the input images and the number of singular values retained.

## Applications of SVD

- Image compression
- Image denoising and noise reduction
- Dimensionality reduction
- Data analysis
- Feature extraction
- Recommendation systems

## Conclusion

This project demonstrates how Singular Value Decomposition connects linear algebra with practical image-processing applications. By decomposing image matrices into singular vectors and singular values, SVD enables image approximation using fewer components and can help reduce certain types of noise. The project also illustrates the geometric interpretation of SVD through rotation, reflection, and scaling.

## Contributors

- **Member 1:** Mathematical Foundation
- **Member 2:** Geometric Interpretation
- **Member 3:** Image Compression
- **Member 4:** Noise Filtering

## References

- [NumPy Documentation — Singular Value Decomposition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [OpenCV Documentation](https://docs.opencv.org/)
