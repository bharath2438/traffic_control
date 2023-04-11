import cv2

bg = cv2.imread("C:/Users/win10/Desktop/test.png")

fg = cv2.imread("C:/Users/win10/AppData/Local/Temp/Traffic3D_Screenshots/2023-04-07_23_25_27_332650/1_shot3.png")

im = cv2.subtract(fg, bg)

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(im, 50, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours = ", len(contours))

h, w = thresh.shape

left = thresh[:, :w//2]

cv2.imshow("image", left)
