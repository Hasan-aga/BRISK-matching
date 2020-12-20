import numpy as np
import cv2

def homography(img1, img2, src_pts, dst_pts, kp1, kp2, good):
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    matchesMask = mask.ravel().tolist()

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    draw_params = dict( # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
    return img3