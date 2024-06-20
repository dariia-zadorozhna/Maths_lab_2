import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# first task
def eigenvalues_and_eigenvectors(matrix):

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # checking if A⋅v=λ⋅v is true
    verification = []
    for i in range(len(eigenvalues)):
        lambda_vector = eigenvalues[i]*eigenvectors[:, i]
        matrix_vector = np.dot(matrix, eigenvectors[:, i])
        verification.append(np.allclose(matrix_vector, lambda_vector))

    return eigenvalues, eigenvectors, verification


def pca_image(n_components):
    pca = PCA(n_components=n_components)
    pca.fit(image_bw)
    trans_pca_image = pca.transform(image_bw)
    reconstructed_image = pca.inverse_transform(trans_pca_image)
    return reconstructed_image, pca


def output_image(image, ax, title):
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title(title)


if __name__ == "__main__":
    matrix = np.array([[1, 2], [3, 4]])
    image = imread("SpongeBob.jpg")
    eigenvalues, eigenvectors, verification = eigenvalues_and_eigenvectors(matrix)
    print("Eigenvalues of the given matrix:\n", eigenvalues)
    print("Eigenvectors of the given matrix:\n", eigenvectors)
    print("Verification:\n", verification)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    print("Image shape is:\n", image.shape)

    image_sum = image.sum(axis=2)
    print("Image_sum shape is:\n", image_sum.shape)
    image_bw = image_sum/image_sum.max()  # This scales all pixel values to the range [0, 1].
    print("Image_bw max is:\n", image_bw.max())

    image_1, pca = pca_image(300)
    plt.imshow(image_1, cmap='gray')
    plt.axis('off')
    plt.show()

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print("Number of components to cover 95% variance:", n_components_95)

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance explained by the components')
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))

    components = [5, 15, 25, 75, 100, 170]
    titles = [f'{c} components' for c in components]

    for ax, n_comp, title in zip(axes.flatten(), components, titles):
        image, _ = pca_image(n_comp)
        output_image(image, ax, title)

    plt.tight_layout()
    plt.show()


