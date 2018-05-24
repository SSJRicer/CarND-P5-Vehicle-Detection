import time
import glob
import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def conv_color(img, color_space='RGB', read_lib='MPL'):
    """
    Converts an image to a chosen color space.

    :param img: A given image.
    :param color_space: Chosen color space (RGB/HSV/LUV/HLS/YUV/YCrCb).
    :param read_lib: Library used for reading the image (matplotlib "MPL"/opencv "CV").

    :return: cvt_img: The converted image.
    """
    if read_lib == 'MPL':  # MatPlotLib (RGB)
        if color_space != 'RGB':
            if color_space == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            cvt_img = np.copy(img)

    else:  # OpenCV (BGR)
        if color_space != 'RGB':
            if color_space == 'HSV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            cvt_img = np.copy(img)

    return cvt_img

# Define a function to return HOG features and visualization:
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    spat_features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return spat_features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately:
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector:
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the feature vector:
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []

        # # (MPL) Read in each one by one:
        # if file.split('.')[-1] == 'png':
        #     image = (mpimg.imread(file)*255).astype(np.uint8)
        # else:
        #     image = mpimg.imread(file)

        # (CV2) Read in each one by one:
        image = cv2.imread(file)

        # apply color conversion if other than 'RGB':
        feature_image = conv_color(image, color_space, read_lib='CV')

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    # Return list of feature vectors:
    return features

########################### TRAINING ################################
# # Creating the car and non-car datasets:
car_img_filenames = glob.glob('datasets/vehicles/**/*.png')
noncar_img_filenames = glob.glob('datasets/non-vehicles/**/*.png')
# car_img_filenames = glob.glob('datasets/vehicles_smallset/**/*.jpeg')
# noncar_img_filenames = glob.glob('datasets/non-vehicles_smallset/**/*.jpeg')
cars_len = len(car_img_filenames)
noncars_len = len(noncar_img_filenames)
print('Number of car images:', cars_len)
print('Number of non-car images:', noncars_len)

model_filename = 'model_bigset_v2.p'

# Parameters for tuning:
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

################################################ RESULTS ###################################################
# Color,   spat,   hist,   orient,   pix,   cell,   hog   ->   acc,   extract,   train
# RGB,     32,     32,     9,        8,     2,      0,         97.58  63.21      13.23
# RGB,     32,     32,     9,        8,     2,      ALL,       97.8   96.58      24.55
# RGB,     32,     32,     11,       16,    2,      ALL,       97.83  53.6       11.14
# YCrCb,   32,     32,     11,       16,    2,      ALL,       98.9   53.83      8.77
# YCrCb,   32,     32,     11,       16,    1,      ALL,       98.42  55.48      7.58
# YCrCb,   32,     32,     11,       32,    2,      ALL,       98.14  40.98      8.09
# YUV,     -,      -,      11,       16,    2,      ALL,       97.78  47.8       3.26
# HLS,     16,     16,     9,        8,     2,      1,         97.38  33.96      5.9
# HLS,     16,     16,     9,        8,     2,      1,         98.49  5.29       0.21    SMALLSET
# HLS,     16,     16,     9,        8,     2,      1,         97.97  37.22      5.68    BIGSET
# YUV,     -,      -,      11,       16,    2,      ALL,       97.38  49.17      2.03    BIGSET
# YCrCb,   16,     16,     12,       16,    2,      ALL,       98.99  56.47      2.94    BIGSET

# HLS,     -,      -,      9,      8,      2,      ALL -> 0.9778, 29.43 to train, 132.31 to extract
# HLS,     12,     16,     2, 1 -> 0.964, 15.5 to train, 71.93 to extract
# YUV, 10, ,32, 2, ALL -> 0.9631, 2.41 to train, 45.04 to extract NO SPAT/HIST
# HSV, 9, 8, 2, ALL -> 0.9572, 26.71 to train, 116 to extract NO SPAT/HIST
# HSV, 9, 8, 2, ALL -> 0.9716, 23.55 to train, 141.47 to extract NO SPAT
############################################################################################################

t = time.time()

car_features = extract_features(car_img_filenames, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

noncar_features = extract_features(noncar_img_filenames, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, noncar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler:
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X:
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', orient, 'orientations,', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

# Saving to a pickle file:
model_pickle = {}
model_pickle['svc'] = svc
model_pickle['scaler'] = X_scaler
model_pickle['color_space'] = color_space
model_pickle['spatial_size'] = spatial_size
model_pickle['hist_bins'] = hist_bins
model_pickle['orient'] = orient
model_pickle['pix_per_cell'] = pix_per_cell
model_pickle['cell_per_block'] = cell_per_block
model_pickle['hog_channel'] = hog_channel
model_pickle['spatial_feat'] = spatial_feat
model_pickle['hist_feat'] = hist_feat
model_pickle['hog_feat'] = hog_feat
pickle.dump(model_pickle, open(model_filename, 'wb'))
