import numpy as np
import cv2 as cv
def explore_match(win, img1, img2, kp1, kp2, good, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2


    # vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    matchesMask = status.ravel().tolist()
    pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    img2new = cv.polylines(img2, [np.int32(dst)], True,255,3, cv.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2new,kp2,good,None,**draw_params)
    cv.imshow("!", img3)
    cv.waitKey()