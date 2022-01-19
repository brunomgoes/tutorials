import matplotlib.pyplot as plt
import numpy as np
from skimage import data, morphology

#get image from skimage data
caller = getattr(data, 'shepp_logan_phantom')
img = caller()

# MORPHOLOGICAL FEATURES
# dilation 
# -> foreground regions grow in size, and foreground features tend to connect or merge 
# -> background features or holes inside a foreground region shrink and sharp corners are smoothed
# -> if at least one pixel in the se coincides with a foreground pixel in the image, 
# then set the output pixel in a new image to the foreground value
# -> dilation with a 3 × 3 se is able to bridge gaps of up to two pixels in length
se =  np.ones((5, 5)) #create the structuring element
im_d = morphology.dilation(img, footprint=se)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow(im_d, cmap='gray')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow((img-im_d), cmap='gray') #original image - dilation
plt.show()

#erosion
# -> increases the size of background objects / shrinks the foreground objects 
# -> if at least one pixel in the se coincides with a background pixel, 
# then the output pixel is set to the background value
# -> foreground features tend to disconnect or further separate
# -> background features or holes inside a foreground region grow and corners are sharpened
# -> it eliminates irrelevant detail, below se's size, in an image
im_e = morphology.erosion(img, footprint=se)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow(im_e, cmap='gray')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow((im_e-img), cmap='gray') #erosion - original image
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow((img-im_d), cmap='gray') #original image - dilation
ax2.imshow((im_e-img), cmap='gray') #erosion - original image
plt.show()

# opening
# -> is an erosion followed by dilation using the same se for both operations
# -> erosion removes some foreground (bright) pixels from the edges of regions 
# of foreground pixels, while dilation adds foreground pixels
# -> the size of the features remain about the same size, but their contours are smoother
# -> narrow isthmuses are broken and thin protusions eliminated
# -> it tends to eliminate foreground regions dissimilar to the se
im_o = morphology.opening(img, footprint=morphology.disk(5))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow(im_o, cmap='gray') #erosion - original image
plt.show()

# closing
im_c = morphology.closing(img, footprint=morphology.disk(5))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax2.imshow(im_c, cmap='gray') #erosion - original image
plt.show()