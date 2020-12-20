import cv2


def ratioFilter(img1, kp1, img2, kp2,matches):
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    return match_img, good