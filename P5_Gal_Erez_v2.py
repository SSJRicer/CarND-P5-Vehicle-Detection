import time
import os
import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

# Folder paths:
test_imgs_path = 'test_images/'
output_imgs_path = 'output_images/'

def plot2images(img, proc_img, proc_title, output_filename, img_title='Original',
                is_img_gray=False, is_proc_gray=False):
    """
    Plots an image before and after procession.

    :param img: A given image.
    :param proc_img: The image after processing it.
    :param proc_title: Processed image's axis title.
    :param output_filename: Name of the saved file.
    :param img_title: A given image's axis title.
    :param is_img_gray: TRUE/FALSE on whether the given image is grayscaled/single channel.
    :param is_proc_gray: TRUE/FALSE on whether the proc_img is grayscaled/single channel.

    :return: NOTHING.
    """
    # # Convert from BGR (OpenCV's format) to RGB (matplotlib's format):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # If the processed image isn't single channeled:
    # if not is_gray:
    #     proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    # Plotting - img vs proc_img:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    # Given image:
    if is_img_gray:
        ax1.imshow(img, cmap='gray', interpolation='nearest')
    else:
        ax1.imshow(img, interpolation='nearest')
    ax1.set_title(img_title, fontsize=25)

    # Processed image:
    if is_proc_gray:
        ax2.imshow(proc_img, cmap='gray', interpolation='nearest')
    else:
        ax2.imshow(proc_img, interpolation='nearest')
    ax2.set_title(proc_title, fontsize=25)

    # Saving the figure:
    fig.savefig(output_imgs_path + output_filename)

    plt.show()

# Creating the car and non-car datasets:
car_img_filenames = glob.glob('datasets/vehicles/**/*.png')
noncar_img_filenames = glob.glob('datasets/non-vehicles/**/*.png')
cars_len = len(car_img_filenames)
noncars_len = len(noncar_img_filenames)
print('Number of car images:', cars_len)
print('Number of non-car images:', noncars_len)

# # Plotting random car images:
# car_fig, car_axes = plt.subplots(5, 4, figsize=(4, 5))
# car_fig.suptitle('Cars', fontsize=25)
# car_axes = car_axes.ravel()
# for i in range(20):
#     car_img = cv2.imread(car_img_filenames[np.random.randint(0, cars_len)])
#     car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
#     car_axes[i].axis('off')
#     car_axes[i].imshow(car_img)
# car_fig.savefig(output_imgs_path + 'rand_cars_plot.png')
#
# # Plotting random non-car images:
# noncar_fig, noncar_axes = plt.subplots(5, 4, figsize=(4, 5))
# noncar_fig.suptitle('Non-cars', fontsize=25)
# noncar_axes = noncar_axes.ravel()
# for i in range(20):
#     noncar_img = cv2.imread(noncar_img_filenames[np.random.randint(0, noncars_len)])
#     noncar_img = cv2.cvtColor(noncar_img, cv2.COLOR_BGR2RGB)
#     noncar_axes[i].axis('off')
#     noncar_axes[i].imshow(noncar_img)
# noncar_fig.savefig(output_imgs_path + 'rand_noncars_plot.png')
#
# plt.show()

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
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def plot_hog(img, fig_title, output_filename, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=True):
    img = conv_color(img, color_space)

    _, hog_1st_ch = get_hog_features(img[:, :, 0], orient=orient,
                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                     vis=vis, feature_vec=feature_vec)
    _, hog_2nd_ch = get_hog_features(img[:, :, 1], orient=orient,
                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                     vis=vis, feature_vec=feature_vec)
    _, hog_3rd_ch = get_hog_features(img[:, :, 2], orient=orient,
                                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                     vis=vis, feature_vec=feature_vec)

    hog_fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    hog_fig.suptitle(fig_title, fontsize=25)
    # hog_fig.tight_layout()
    axes = axes.ravel()

    axes[0].imshow(img[:, :, 0], cmap='gray', interpolation='nearest')
    axes[0].set_title(color_space[0:1] + '-CH', fontsize=10)
    axes[1].imshow(hog_1st_ch, cmap='gray', interpolation='nearest')
    axes[1].set_title('Hog ' + color_space[0:1] + '-CH', fontsize=10)

    axes[2].imshow(img[:, :, 1], cmap='gray', interpolation='nearest')
    axes[2].set_title(color_space[1:2] + '-CH', fontsize=10)
    axes[3].imshow(hog_2nd_ch, cmap='gray', interpolation='nearest')
    axes[3].set_title('Hog ' + color_space[1:2] + '-CH', fontsize=10)

    axes[4].imshow(img[:, :, 2], cmap='gray', interpolation='nearest')
    axes[4].set_title(color_space[2:3] + '-CH', fontsize=10)
    axes[5].imshow(hog_3rd_ch, cmap='gray', interpolation='nearest')
    axes[5].set_title('Hog ' + color_space[2:3] + '-CH', fontsize=10)

    # Saving the figure:
    hog_fig.savefig(output_imgs_path + output_filename)
    plt.show()

# car_img = mpimg.imread(car_img_filenames[np.random.randint(0, cars_len)])
# noncar_img = mpimg.imread(noncar_img_filenames[np.random.randint(0, noncars_len)])

# plot_hog(car_img, fig_title='Car [RGB]', output_filename='car_rgb_hog.png', color_space='RGB')
# plot_hog(car_img, fig_title='Car [HSV]', output_filename='car_hsv_hog.png', color_space='HSV')
# plot_hog(car_img, fig_title='Car [LUV]', output_filename='car_luv_hog.png', color_space='LUV')
# plot_hog(car_img, fig_title='Car [HLS]', output_filename='car_hls_hog.png', color_space='HLS')
# plot_hog(car_img, fig_title='Car [YUV]', output_filename='car_yuv_hog.png', color_space='YUV')
# plot_hog(car_img, fig_title='Car [YCrCb]', output_filename='car_YCrCb_hog.png', color_space='YCrCb')
#
# plot_hog(noncar_img, fig_title='Non-Car [RGB]', output_filename='noncar_rgb_hog.png', color_space='RGB')
# plot_hog(noncar_img, fig_title='Non-Car [HSV]', output_filename='noncar_hsv_hog.png', color_space='HSV')
# plot_hog(noncar_img, fig_title='Non-Car [LUV]', output_filename='noncar_luv_hog.png', color_space='LUV')
# plot_hog(noncar_img, fig_title='Non-Car [HLS]', output_filename='noncar_hls_hog.png', color_space='HLS')
# plot_hog(noncar_img, fig_title='Non-Car [YUV]', output_filename='noncar_yuv_hog.png', color_space='YUV')
# plot_hog(noncar_img, fig_title='Non-Car [YCrCb]', output_filename='noncar_YCrCb_hog.png', color_space='YCrCb')


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    spat_features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return spat_features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = conv_color(image, color_space)

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
    # Return list of feature vectors
    return features

# # Parameters for tuning:
# color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9
# pix_per_cell = 8
# cell_per_block = 2
# hog_channel = 'ALL'
# spatial_size = (16, 16)
# hist_bins = 32
#
# #####
# # HLS, 9, 8, 2, ALL -> 0.9778, 29.43 to train, 132.31 to extract
# # HLS, 12, 16, 2, 1 -> 0.964, 15.5 to train, 71.93 to extract
# # YUV, 10, ,32, 2, ALL -> 0.9631, 2.41 to train, 45.04 to extract NO SPAT/HIST
# # HSV, 9, 8, 2, ALL -> 0.9572, 26.71 to train, 116 to extract NO SPAT/HIST
# # HSV, 9, 8, 2, ALL -> 0.9716, 23.55 to train, 141.47 to extract NO SPAT
# #####
#
# t = time.time()
# car_features = extract_features(car_img_filenames, color_space=color_space, orient=orient,
#                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                 hog_channel=hog_channel, spatial_feat=False, hist_feat=False)
#
# notcar_features = extract_features(noncar_img_filenames, color_space=color_space, orient=orient,
#                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                                    hog_channel=hog_channel, spatial_feat=False, hist_feat=False)
# t2 = time.time()
# print(round(t2 - t, 2), 'Seconds to extract HOG features...')
#
# # Create an array stack of feature vectors
# X = np.vstack((car_features, notcar_features)).astype(np.float64)
#
# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
#
# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=rand_state)
#
# # Fit a per-column scaler
# X_scaler = StandardScaler().fit(X_train)
# # Apply the scaler to X
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)
#
# print('Using:', orient, 'orientations', pix_per_cell,
#       'pixels per cell and', cell_per_block, 'cells per block')
# print('Feature vector length:', len(X_train[0]))
#
# # Use a linear SVC
# svc = LinearSVC()
# # Check the training time for the SVC
# t = time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2 - t, 2), 'Seconds to train SVC...')
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t = time.time()
# n_predict = 10
# print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
# print('For these', n_predict, 'labels: ', y_test[0:n_predict])
# t2 = time.time()
# print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
#
# # img = mpimg.imread('test_image.jpg')
#
# model_pickle = {}
# model_pickle['svc'] = svc
# model_pickle['scaler'] = X_scaler
# model_pickle['color_space'] = color_space
# model_pickle['orient'] = orient
# model_pickle['pix_per_cell'] = pix_per_cell
# model_pickle['cell_per_block'] = cell_per_block
# model_pickle['spatial_size'] = spatial_size
# model_pickle['hist_bins'] = hist_bins
# pickle.dump(model_pickle, open('model2.p', 'wb'))

if os.path.exists('model3.p'):
    print('Model2 file found')
    with open('model2.p', mode='rb') as f:
        model = pickle.load(f)
    svc = model['svc']
    X_scaler = model['scaler']
    color_space = model['color_space']
    orient = model['orient']
    pix_per_cell = model['pix_per_cell']
    cell_per_block = model['cell_per_block']
    spatial_size = model['spatial_size']
    hist_bins = model['hist_bins']

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = conv_color(img_tosearch, color_space='YCrCb', read_lib='CV')

    # Rescale image if not 1.0 scale:
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # # Extract the image patch
            # subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            #
            # # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=hist_bins)
            #
            # # Scale features and make a prediction
            # test_features = X_scaler.transform(
            #     np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(hog_features.reshape(1, -1))

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img

# # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#         top_left = min_loc
#         bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
#
#         cv2.rectangle(img,top_left, bottom_right, 255, 2)
#         bbox_list.append((top_left, bottom_right))

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

ystart = 400
ystop = 660
scale = 1.5

test_img_filename = "test6.jpg"
test_img = cv2.cvtColor(cv2.imread(test_imgs_path + test_img_filename), cv2.COLOR_BGR2RGB)

out_img = find_cars(test_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

plt.imshow(out_img)
plt.show()

# heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
#
# def add_heat(heatmap, bbox_list):
#     # Iterate through list of bboxes
#     for box in bbox_list:
#         # Add += 1 for all pixels inside each bbox
#         # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#         heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
#
#     # Return updated heatmap
#     return heatmap
#
# def apply_threshold(heatmap, threshold):
#     # Zero out pixels below the threshold
#     heatmap[heatmap <= threshold] = 0
#     # Return thresholded map
#     return heatmap
#
# def draw_labeled_bboxes(img, labels):
#     # Iterate through all detected cars
#     for car_number in range(1, labels[1]+1):
#         # Find pixels with each car_number label value
#         nonzero = (labels[0] == car_number).nonzero()
#         # Identify x and y values of those pixels
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         # Define a bounding box based on min/max x and y
#         bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
#         # Draw the box on the image
#         cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
#     # Return the image
#     return img
#
#
# # Add heat to each box in box list
# heat = add_heat(heat, box_list)
#
# # Apply threshold to help remove false positives
# heat = apply_threshold(heat, 1)
#
# # Visualize the heatmap when displaying
# heatmap = np.clip(heat, 0, 255)
#
# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(test_img), labels)