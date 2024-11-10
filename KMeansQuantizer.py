import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load images
marioBrosOriginal = io.imread("marioAndLuigi.jpg")
marioBrosMovieOriginal = io.imread("marioBrosMovie.jpg")
marioBrosMovie = io.imread("marioBrosMovie.jpg")

# Display original image before quantization
plt.figure(1)
plt.title("marioAndLuigi.jpg ORIGINAL BEFORE quantization")
plt.imshow(marioBrosOriginal)
plt.axis('off')
plt.show()

# Reshape images for KMeans clustering
marioBrosReshaped = marioBrosOriginal.reshape(-1, 3)
marioBrosMovieReshaped = marioBrosMovie.reshape(-1, 3)

# Function to quantize and display an image
def quantize_and_display(image, kmeans, figure_num, k):
    quantized_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
    plt.figure(figure_num)
    plt.title(f"marioAndLuigi.jpg - K={k}")
    plt.imshow(quantized_image.astype(np.uint8))
    plt.axis('off')
    plt.show()
    print(f"k = {k}, SSE = {kmeans.inertia_}")
    return kmeans.inertia_

# Function for Task 2: Quantize second image with given cluster centers
def quantize_task2(image_reshaped, cluster_centers, labels, original_shape):
    quantized_image = cluster_centers[labels].reshape(original_shape)
    return quantized_image.astype(np.uint8)

# Perform KMeans clustering and plot results
k_values = range(2, 20, 2)
inertias = []
best_k = 10  # Chosen based on visual results

for idx, k in enumerate(k_values, start=2):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(marioBrosReshaped)
    inertias.append(quantize_and_display(marioBrosOriginal, kmeans, idx, k))

# Plot inertia (elbow method) graph
plt.figure(len(k_values) + 2)
plt.title("marioAndLuigi.jpg k vs. inertia")
plt.plot(k_values, inertias, marker='o')
plt.xlabel('k')
plt.ylabel('Inertia (SSE)')
plt.show()

# Quantize Task 2 image using best k
best_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(marioBrosReshaped)
task2_labels = best_kmeans.predict(marioBrosMovieReshaped)
marioBrosMovieQuantized = quantize_task2(marioBrosMovieReshaped, best_kmeans.cluster_centers_, task2_labels, marioBrosMovie.shape)

# Display quantized and original images for Task 2
plt.figure(len(k_values) + 3)
plt.title(f"marioBrosMovie.jpg quantized using marioAndLuigi.jpg K={best_k}")
plt.imshow(marioBrosMovieQuantized)
plt.axis('off')
plt.show()

plt.figure(len(k_values) + 4)
plt.title("marioBrosMovie.jpg ORIGINAL BEFORE quantization")
plt.imshow(marioBrosMovieOriginal)
plt.axis('off')
plt.show()
