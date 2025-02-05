
import cv2 
import numpy as np 
import matplotlib.pyplot as plt


filename = 'images/img2.jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

harris=img.copy()
# Threshold for an optimal value, it may vary depending on the image.
harris[dst>0.01*dst.max()]=[0,0,255]

# Display the images
plt.figure(figsize=(12, 9))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the generated output image for Harris
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(harris, cv2.COLOR_BGR2RGB))
plt.title('Output Image (Harris)')
plt.axis('off')

plt.tight_layout()
plt.show()
