# Advance lane lines project
# Neil Maude
# February 2017

# Tests of the various functions, used to check the project during build
# Should all run, effectively a unit test of the project components

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import advanced_lane_lines as AdvLanes

# Part 1: Calibration
# Calibration testing
# This code takes the provided calibration images and calibrates the camera
calibrate_samples_dir = 'camera_cal'
calibrate_pattern = 'calibration*.jpg'
mtx, dist = AdvLanes.calibrate_camera(calibrate_samples_dir, calibrate_pattern, fVerbose=True)

#Part 2: Removing distortion
# Test undistort of images
test_samples_dir = 'test_images'
test_samples_pattern1 = 'straight_lines*.jpg'
test_samples_pattern2 = 'test*.jpg'
verbose_dir = 'verbose_output'

# Test one of the calibration examples, to properly test the undistort call
# Image #2 has obvious distortion, so easy to see that the calibration has worked out OK
img = cv2.imread(calibrate_samples_dir +'/calibration2.jpg')
undist = AdvLanes.undistort_image(img,mtx,dist)
cv2.imwrite(verbose_dir + '/test_calibration.jpg', undist)

# Now just run over all of the test images, straight lines and general tests, outputting those to the verbose dir
images = glob.glob(test_samples_dir + '/' + test_samples_pattern1)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)  # Note: will be in BGR form
    undist = AdvLanes.undistort_image(img, mtx, dist)
    # write out the undistorted file
    fOutfilename = 'sl_undistort' + str(idx+1)
    AdvLanes.create_dir(verbose_dir)
    print('Creating undistorted file: ', fOutfilename + '.jpg')
    output_img_name = verbose_dir + '/' + fOutfilename + '.jpg'
    cv2.imwrite(output_img_name, undist)
images = glob.glob(test_samples_dir + '/' + test_samples_pattern2)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)  # Note: will be in BGR form
    undist = AdvLanes.undistort_image(img, mtx, dist)
    # write out the undistorted file
    fOutfilename = 'test_undistort' + str(idx+1)
    AdvLanes.create_dir(verbose_dir)
    print('Creating undistorted file: ', fOutfilename + '.jpg')
    output_img_name = verbose_dir + '/' + fOutfilename + '.jpg'
    cv2.imwrite(output_img_name, undist)

# Part 3: create some binary images using the Sobel functions and HLS space calculations
# This is going to have some iterations
# The objective is to find some settings which work well to find lines in all lighting conditions
# Should get some decent results on each of the sample images (straight lines and tests)

# Example - take the sobel directional gradient in the y-direction for each test image
# These tests output a messy image, but test the process
ksize = 3 # Choose a larger odd number to smooth gradient measurements
images = glob.glob(test_samples_dir + '/' + test_samples_pattern2)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)  # Note: will be in BGR form
    # have an image, now take the directional sobel threshold
    # some magic numbers for the thresholds, which seem to work OK
    dir_binary = AdvLanes.sobel_dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    #dir_binary = thresholds.sobel_dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi))
    #dir_binary = thresholds.sobel_abs_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # write out the directional gradient threshold image
    fOutfilename = 'test_dir_thresh' + str(idx+1)
    AdvLanes.create_dir(verbose_dir)
    print('Creating directional threshold file: ', fOutfilename + '.png')
    output_img_name = verbose_dir + '/' + fOutfilename + '.png'
    im_bw = AdvLanes.mask2binary(dir_binary)    # convert to a binary image
    cv2.imwrite(output_img_name, im_bw)

# Example - s-channel thresholding
thresh = (90, 255)      # magic numbers
images = glob.glob(test_samples_dir + '/' + test_samples_pattern2)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)  # Note: will be in BGR form
    # have an image, now take the directional sobel threshold
    img_binary = AdvLanes.hls_s_channel(img, thresh)
    # write out the directional gradient threshold image
    fOutfilename = 'test_s-channel_thresh' + str(idx+1)
    AdvLanes.create_dir(verbose_dir)
    print('Creating s-channel threshold file: ', fOutfilename + '.png')
    output_img_name = verbose_dir + '/' + fOutfilename + '.png'
    im_bw = AdvLanes.mask2binary(img_binary)    # convert to a binary image
    cv2.imwrite(output_img_name, im_bw)

# Example - combined thresholding
# grad_x and hls s-channel
images = glob.glob(test_samples_dir + '/' + test_samples_pattern2)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)  # Note: will be in BGR form
    # have an image, now take the directional sobel threshold
    s_binary = AdvLanes.hls_s_channel(img, (90, 255))
    g_binary = AdvLanes.sobel_abs_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    combined_binary = AdvLanes.combine_mask(s_binary, g_binary)
    # write out the directional gradient threshold image
    fOutfilename = 'test_combined_thresh' + str(idx+1)
    AdvLanes.create_dir(verbose_dir)
    print('Creating combined threshold file: ', fOutfilename + '.png')
    output_img_name = verbose_dir + '/' + fOutfilename + '.png'
    im_bw = AdvLanes.mask2binary(combined_binary)    # convert to a binary image
    cv2.imwrite(output_img_name, im_bw)

# Part 4: warping of the image
# test of general warping - write out warped versions of each of the test images
test_samples_dir = 'test_images'
test_samples_pattern2 = 'test*.jpg'
src = np.float32([[235,700], [580,460], [700,460], [1070,700]])
dst = np.float32([[320,720], [320,0], [960,0], [960,720]])
images = glob.glob(test_samples_dir + '/' + test_samples_pattern2)
mtx, dist = AdvLanes.load_calibration_data()
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img_undist = AdvLanes.undistort_image(img, mtx, dist)
    warped = AdvLanes.warp_image(img_undist, src, dst)
    cv2.line(warped, (320, 720), (320, 0), color=[255, 0, 0], thickness=5)
    cv2.line(warped, (960, 720), (960, 0), color=[255, 0, 0], thickness=5)
    cv2.imwrite(verbose_dir + '/' + 'test_warped' + str(idx+1) + '.jpg', warped)

# Part 5: Lane finding
# test of polynominal finding - find the polynomial for a test image
img_binary = AdvLanes.preferred_threshold_image_from_file('test_images/test1.jpg')            # create a binary image
binary_warped = AdvLanes.warp_original_to_overhead(img_binary)                      # warp it using pipeline defaults
left_fit, right_fit, leftx, lefty, rightx, righty, out_img = AdvLanes.find_lanes_sliding(binary_warped)           # get the polynominal fit

# code below is to draw the polynominal lines on the image and output an example
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
f, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(out_img)
ax.plot(left_fitx, ploty, color='yellow')
ax.plot(right_fitx, ploty, color='yellow')
f.savefig(verbose_dir + '/sliding_result_test1.png')
plt.close()

# now test the finding of a polynomial when already have a line fit, by searching around the lines
left_fit, right_fit, leftx, lefty, rightx, righty = AdvLanes.find_lanes_search(binary_warped, left_fit, right_fit)    # takes the previous polys

# code below is to draw the search zone and output an example
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-AdvLanes.MARGIN_SLIDING_WINDOWS, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+AdvLanes.MARGIN_SLIDING_WINDOWS, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-AdvLanes.MARGIN_SLIDING_WINDOWS, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+AdvLanes.MARGIN_SLIDING_WINDOWS, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))
# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
f, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(result)
ax.plot(left_fitx, ploty, color='yellow')
ax.plot(right_fitx, ploty, color='yellow')
f.savefig(verbose_dir + '/sliding_search_test1.png')
plt.close()

# code below is to generate a polygon and lane lines a blank binary warped
result = AdvLanes.plot_lanes_on_warped(np.zeros_like(binary_warped), left_fit, right_fit)
f, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(result)
f.savefig(verbose_dir + '/lane_marked.png')
plt.close()

# Now warp the binary image back to the original perspective
unwarped = AdvLanes.warp_overhead_to_original(result)
f, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(unwarped)
f.savefig(verbose_dir + '/sliding_lane_marks_unwarped.png')
plt.close()

# code below overlays the line polygon over the original image
original_img = cv2.imread('test_images/test1.jpg')
final_img = AdvLanes.merge_over_camera_view(original_img, unwarped)
f, ax = plt.subplots( nrows=1, ncols=1 )
ax.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
f.savefig(verbose_dir + '/sliding_lane_marks_final.png')
plt.close()

# code below gets the radius for an image frame (using the points detected)
radius = AdvLanes.get_curvature(leftx, lefty, rightx, righty)
print(radius, 'm')

# code below gets the offset for an image frame (using the points detected)
offset = AdvLanes.get_offset(original_img.shape[1], left_fit, right_fit, leftx, lefty, rightx, righty)
print('Calculated offset (mtrs): ', offset)

# create a final example image
image_final = AdvLanes.add_text(final_img, radius[0], offset)
cv2.imwrite(verbose_dir + '/' + 'final_image.jpg', image_final)