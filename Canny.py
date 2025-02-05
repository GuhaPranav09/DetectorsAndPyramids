
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
img = cv2.imread('images/img.jpeg')

# Convert the original image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Generate Canny edge detection image
canny_e = cv2.Canny(gray_image, 100, 150)

# Define the random constant c
c = 1

# Perform the element-wise addition for Canny
g_canny = cv2.add(img, cv2.merge([canny_e] * 3) * c)
g_canny = np.clip(g_canny, 0, 255).astype(np.uint8)

# Display the images
plt.figure(figsize=(12, 9))

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the generated Canny edge detection image
plt.subplot(1, 3, 2)
plt.imshow(canny_e, cmap='gray')
plt.title('Generated Canny Edge Detection')
plt.axis('off')

# Display the generated output image for Canny
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(g_canny, cv2.COLOR_BGR2RGB))
plt.title('Output Image (Canny)')
plt.axis('off')

plt.tight_layout()
plt.show()
