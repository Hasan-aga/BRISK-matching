import numpy as np
import cv2 as cv
def explore_match(win, img1, img2, kp1, kp2, good, status = None, H = None):
    # draw homography and lines between matches
    h1, w1 = img1.shape[:2]
    matchesMask = status.ravel().tolist()
    pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    img2new = cv.polylines(img2, [np.int32(dst)], True,255,3, cv.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2new,kp2,good,None,**draw_params)
    return img3