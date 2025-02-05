import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load images
A = cv.imread('images/apple.png')
B = cv.imread('images/orange.png')
assert A is not None, "Apple image file could not be read."
assert B is not None, "Orange image file could not be read."

A = cv.cvtColor(A, cv.COLOR_BGR2RGB)
B = cv.cvtColor(B, cv.COLOR_BGR2RGB)

# Generate Gaussian pyramids
gpA = [A.copy()]
gpB = [B.copy()]
for i in range(6):
    gpA.append(cv.pyrDown(gpA[-1]))
    gpB.append(cv.pyrDown(gpB[-1]))

# Generate Laplacian pyramids
lpA = [gpA[5]]
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE_A = cv.pyrUp(gpA[i], dstsize=(gpA[i-1].shape[1], gpA[i-1].shape[0]))
    lpA.append(cv.subtract(gpA[i-1], GE_A))
    
    GE_B = cv.pyrUp(gpB[i], dstsize=(gpB[i-1].shape[1], gpB[i-1].shape[0]))
    lpB.append(cv.subtract(gpB[i-1], GE_B))

# Blend pyramids
LS = []
for la, lb in zip(lpA, lpB):
    cols = la.shape[1]
    ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# Reconstruct blended image
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_, dstsize=(LS[i].shape[1], LS[i].shape[0]))
    ls_ = cv.add(ls_, LS[i])

# Direct blending
real = np.hstack((A[:, :A.shape[1] // 2], B[:, B.shape[1] // 2:]))

# Convert blended images to RGB for display
ls_ = np.clip(ls_, 0, 255).astype(np.uint8)

# Plot results
fig, axes = plt.subplots(4, 6, figsize=(15, 10))

# Display Gaussian pyramids
for i in range(6):
    axes[0, i].imshow(gpA[i])
    axes[0, i].set_title(f'Gaussian A {i}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(gpB[i])
    axes[1, i].set_title(f'Gaussian B {i}')
    axes[1, i].axis('off')

# Display Laplacian pyramids
for i in range(6):
    axes[2, i].imshow(cv.convertScaleAbs(lpA[i]))
    axes[2, i].set_title(f'Laplacian A {i}')
    axes[2, i].axis('off')
    
    axes[3, i].imshow(cv.convertScaleAbs(lpB[i]))
    axes[3, i].set_title(f'Laplacian B {i}')
    axes[3, i].axis('off')

plt.show()

# Show final images
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0][0].imshow(A)
ax[0][0].set_title('Original Apple')
ax[0][0].axis('off')

ax[0][1].imshow(B)
ax[0][1].set_title('Original Orange')
ax[0][1].axis('off')

ax[1][0].imshow(real)
ax[1][0].set_title('Direct Blending')
ax[1][0].axis('off')

# Show blended result
ax[1][1].imshow(ls_)
ax[1][1].set_title('Pyramid Blending')
ax[1][1].axis('off')

plt.show()
