import cv2

# read in image to detect pickle ball by providing image path.
photo = cv2.imread("C:\CS\personal projects\image recognition\pickleball.jpg")

# image is converted to gray-scale to perfum calculates for Canny Edge detection via a single-channel.
gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
# image is blurred to remove noise which could interfere with Canny Edge detection.
blurred_photo = cv2.GaussianBlur(gray_photo, (9, 9), 2)
# Canny edge detection uses local maximums of gradients to find edges.
edges = cv2.Canny(blurred_photo, 50, 250)
# using edges detected, the most probable circles that fit the edges are found using a 
# tally-system and testing different radii & centers.
circles = cv2.HoughCircles(
    blurred_photo, cv2.HOUGH_GRADIENT, dp=1.1, minDist=100, param1=100, param2=50, minRadius=10, maxRadius=100
)
# circles found using HoughCircles, are outlines in neon green to show detected pickleballs
if circles is not None:
    circles = circles[0, :].astype(int)
    for (x, y, r) in circles:
        cv2.circle(photo, (x, y), r, (0, 255, 0), 4)
# image with detected pickleballs is shown.
cv2.imshow('Detected Pickleball', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
