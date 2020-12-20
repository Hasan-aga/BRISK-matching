import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from imports.filter_matches import filter_matches
from imports.explore_match import explore_match
from imports.homography import homography
from imports.normalfilter import ratioFilter


# path of image 1
path1 = "./images/img1.jpg"

# path of image 2
path2 = "./images/img2.jpg"

img1 = cv.imread(path1) 
img2 = cv.imread(path2) 

# img1 = cv.cvtColor(inIMG1, cv.COLOR_GRAY2BGR)
# img2 = cv.cvtColor(inIMG2, cv.COLOR_GRAY2BGR)
# cv.imshow("what", img1)
# cv.waitKey()

detector = cv.BRISK_create()
norm = cv.NORM_HAMMING
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)
result1 = cv.drawKeypoints(img1, kp1, None)
cv.imwrite("features1.jpg", result1)
result2 = cv.drawKeypoints(img2, kp2, None)
cv.imwrite("features2.jpg", result2)

FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
matcher = cv.FlannBasedMatcher(flann_params, {})
raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
p1, p2, kp_pairs, good = filter_matches(kp1, kp2, raw_matches)
if len(p1) >= 4:
    H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
    print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
else:
    H, status = None, None
    print('%d matches found, not enough for homography estimation' % len(p1))
vis = explore_match("win", img1, img2, kp1, kp2, good, status, H)
cv.imwrite("match.jpg", vis)
# 
