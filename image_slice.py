import cv2
import matplotlib.pyplot as plt
  
img = cv2.imread('C:/Users/win10/Desktop/1_shot27.png')
bg = cv2.imread('C:/Users/win10/Desktop/test.png')
def getCroppedImage(img):
    rows = img.shape[0]
    cols = img.shape[1]
    offsetX = round(0.01768*rows)
    offsetY = round(0.2391*cols)
    return img[offsetX:rows-offsetX, offsetY:cols-offsetY]

imgCropped = getCroppedImage(img)
bgCropped = getCroppedImage(bg)

imgResized = cv2.resize(imgCropped, (200, 200))
bgResized = cv2.resize(bgCropped, (200, 200))

finalImg = cv2.subtract(imgResized, bgResized)
cv2.imwrite('C:/Users/win10/Desktop/final.png', finalImg)

imageplot = plt.imshow(finalImg)
plt.show()

##h, w, channels = img.shape
##  
##half = w//2
##
##left_part = img[int(h/3.5):int(h/1.75), :int(w/2.5)] 
##
##right_part = img[int(h/3.5):int(h/1.75), int(w/1.75):]  
##  
##cv2.imshow('Left part', left_part)
##cv2.imshow('Right part', right_part)
##
##half2 = h//2
##  
##top = img[:h//3, int(w/2.5):int(w/1.75)]
##bottom = img[int(h/1.75):, int(w/3.25):int(w/1.5)]
##  
##cv2.imshow('Top', top)
##cv2.imshow('Bottom', bottom)
