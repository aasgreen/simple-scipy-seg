import numpy as np
import matplotlib.pyplot as plt
import skimage
import sklearn
import pathlib
import pims
import skimage.filters
import sklearn.preprocessing
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter
from sklearn.feature_extraction import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import skimage.exposure
from skimage.morphology import disk
from skimage.transform import rescale


masks = pims.open('./data/seg/*.png')
image = plt.imread('./data/seg/meat-label.png')
#display images so we know we have the right mask

#lines = [plt.imshow(im) for im in masks[::-1]]


#convert rgb to bw (image is already bw, so just converting it makes things simpler)
radius = 200
image = skimage.color.rgb2gray(image)
image_sm = rescale(gaussian_filter(image, sigma = 2), .02, mode = 'reflect', anti_aliasing=False, multichannel = False)
image_hc = skimage.exposure.equalize_hist(image)
image_blur = gaussian_filter(image_hc, sigma = 50)
#image_sm = rescale(gaussian_filter(image_hc, sigma = 2), .02, mode = 'reflect', anti_aliasing=False, multichannel = False)
#image_grad = skimage.filters.scharr(image)
#image_entropy = skimage.filters.rank.entropy(image_hc, disk(radius))

#we will use the intensity and the gradient to do our unsupervised k-means clustering

#first, normalize both the intensity feature and the image_grad feature
scaler = sklearn.preprocessing.MinMaxScaler()
intensity_feature = image_hc.ravel()
#gradient_feature = image_grad.ravel()
#entropy_feature = image_entropy.ravel()
blur_feature = image_blur.ravel()
#first, create design matrix

design = np.c_[intensity_feature, blur_feature]
design_norm = scaler.fit_transform(design)

#load kmeans

#kmeans = KMeans(n_clusters = 4, random_state = 42).fit(design)
#result = kmeans.labels_.reshape(image.shape[0], image.shape[1])
#plt.imshow(result, cmap = 'viridis')
#

#it did a shitty job clustering. I think it's triggering on the gradient too much. Try clustering on the single intensity feature

design_norm_1 = design_norm
kmeans_1 = KMeans(n_clusters = 4, random_state = 42).fit(design_norm_1)
result = kmeans_1.labels_.reshape(image.shape[0], image.shape[1])
plt.imshow(result, cmap = 'viridis')


#try agglomerative clustering
#X = np.reshape(image_sm, (-1,1))
#n_clusters = 5
#connect = grid_to_graph(*image_sm.shape)
#ward = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward', connectivity = connect)
#ward.fit(X)
#labels = np.reshape(ward.labels_, image_sm.shape)
#
##plt.imshow(image_sm)
#plt.imshow(labels, cmap = 'viridis', alpha = .1)
