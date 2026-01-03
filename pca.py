import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#When image is read, it's in BGR form, convert it to RGB. Then print out the original image
img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
print(img.shape)
plt.imshow(img)
plt.show()

#Splitting the 3D array into 3 2D arrays. Each 2D array stores rows x columns, entry is the value ranging from [0,255] of the color
r, g, b = cv2.split(img)
#Normalize the data i.e each value is in range [0,1] required by PCA()
r, g, b = r / 255, g / 255, b / 255

#Choose the # of eigenvectors
pca_components = 300

#Performing PCA on the 3 arrays.
pca_r = PCA(n_components = pca_components)
reduced_r = pca_r.fit_transform(r)

pca_g = PCA(n_components = pca_components)
reduced_g = pca_g.fit_transform(g)

pca_b = PCA(n_components = pca_components)
reduced_b = pca_b.fit_transform(b)

#Merge back the 3 arrays to produce the "reduced" image. However, the image is compressed, hence it doesn't look anywhere alike the original
image_reduced = (cv2.merge((reduced_r, reduced_g, reduced_b)))
image_reduced = np.clip(image_reduced, 0, 1)
#Showing the compressed image
plt.imshow(image_reduced)
plt.show()


#Reconstruct the reduced array back to the original dimension; however, only keeping the variance of the reduced array. Allow us to compare how much 
#variance is lost between the original and the reduced image
reconstructed_r = pca_r.inverse_transform(reduced_r)
reconstructed_g = pca_g.inverse_transform(reduced_g)
reconstructed_b = pca_b.inverse_transform(reduced_b)

image_reconstructed = (cv2.merge((reconstructed_r,reconstructed_g,reconstructed_b)))
image_reconstructed = np.clip(image_reconstructed, 0, 1)
plt.imshow(image_reconstructed)
plt.show()

#Save the image to disk to compare it with different values of pca_components
image_reconstructed_uint8 = (image_reconstructed * 255).astype(np.uint8)
image_reconstructed_bgr = cv2.cvtColor(image_reconstructed_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite("imaged_reconstructed.jpg", image_reconstructed_bgr)