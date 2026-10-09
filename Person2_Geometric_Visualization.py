import numpy as np
import matplotlib.pyplot as plt


def geometric_visualization(A):

    if A.shape != (2, 2):
        print("Error: Please enter a 2x2 matrix.")
        return

    # SVD
    U, S, VT = np.linalg.svd(A)

    # Sigma matrix
    Sigma = np.diag(S)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)

    circle = np.array([
        np.cos(theta),
        np.sin(theta)
    ])

    # Apply V^T
    after_VT = VT @ circle

    # Apply Sigma
    after_Sigma = Sigma @ after_VT

    # Apply U
    after_U = U @ after_Sigma

    # Direct transformation using A
    direct_A = A @ circle

    # Display SVD
    print("\n----------------------------------------")
    print("GEOMETRIC VISUALIZATION OF SVD")
    print("----------------------------------------")

    print("\nMatrix A:")
    print(A)

    print("\nU:")
    print(U)

    print("\nSingular Values:")
    print(S)

    print("\nSigma:")
    print(Sigma)

    print("\nV^T:")
    print(VT)

    # Verify A = U Sigma V^T
    reconstructed_A = U @ Sigma @ VT

    print("\nU Sigma V^T:")
    print(reconstructed_A)

    print("\nVerification:")
    print(np.allclose(A, reconstructed_A))

    # Find graph limits
    all_points = np.concatenate(
        [circle, after_VT, after_Sigma, after_U, direct_A],
        axis=1
    )

    max_value = np.max(np.abs(all_points)) + 0.5

    # Create plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Original Circle
    axes[0].plot(circle[0], circle[1])
    axes[0].set_title("Original Unit Circle")

    # After V^T
    axes[1].plot(after_VT[0], after_VT[1])
    axes[1].set_title("After $V^T$")

    # After Sigma
    axes[2].plot(after_Sigma[0], after_Sigma[1])
    axes[2].set_title("After $\\Sigma$")

    # After U
    axes[3].plot(after_U[0], after_U[1])
    axes[3].set_title("After $U$")

    # Direct transformation
    axes[4].plot(direct_A[0], direct_A[1])
    axes[4].set_title("Direct Transformation by A")

    # Format all graphs
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-max_value, max_value)
        ax.set_ylim(-max_value, max_value)
        ax.grid(True)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

    plt.suptitle("Geometric Interpretation of SVD", fontsize=16)
    plt.tight_layout()
    plt.show()


# Example matrix
A = np.array([
    [2, 1],
    [-1, 1]
])

# Call the function
geometric_visualization(A)