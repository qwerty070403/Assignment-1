#Plotting Keypoints t-SNE(Focuses on analyzing the local features (keypoints) extracted by SIFT.)


# Use OpenCV SIFT to extract descriptors for only 10 images
import cv2
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from main_svm import train_images


  
def extract_sift_descriptors_subset(image_paths):
    sift = cv2.SIFT_create()  # Create a SIFT detector
    all_descriptors = []

    for path in image_paths:
        img = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)  # Convert to grayscale
        keypoints, descriptors = sift.detectAndCompute(img, None)  # Detect keypoints and compute descriptors
        if descriptors is not None:
            all_descriptors.append(descriptors)

    # Combine all descriptors into a single array
    return np.vstack(all_descriptors) if all_descriptors else np.array([])



# Take the first 10 images
subset_image_paths = train_images[:10]  # Replace 'train_images' with your list of image paths
subset_descriptors = extract_sift_descriptors_subset(subset_image_paths)
print(f"Shape of subset descriptors: {subset_descriptors.shape}")



# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(subset_descriptors)
print(f"t-SNE results shape: {tsne_results.shape}")



# Plot the t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', s=10, alpha=0.7)
plt.title('t-SNE Visualization of SIFT Keypoints (10 Images)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
