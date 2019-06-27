import cv2
import numpy as np
import getplate

tam_kernel = 5;
tam_kernel2 = 2;



org = cv2.imread("br/1.jpg")
org1 = getplate.getplate("br/1.jpg")
org1 = cv2.resize(org1,(512,200))
img = cv2.cvtColor(org1, cv2.COLOR_BGR2GRAY)
img = abs(255-img)
kernel = np.ones((tam_kernel,tam_kernel), np.uint8)
img1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
img2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
img = cv2.add(img,img1)
img = cv2.absdiff(img,img2)
img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,-30) # usando 2001 o código fica pesado mas o suficiente para rodar e a definição fica boa o suficiente
img = cv2.GaussianBlur(img,(3,3),0)
a,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((tam_kernel2,tam_kernel2), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
copy = img.copy()
tam_kernel2 = 3;
kernel = np.ones((tam_kernel2,tam_kernel2), np.uint8)
imgt = cv2.dilate(img,kernel,iterations = 2)
contours, hierarchy = cv2.findContours(imgt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
crop_imgs = []
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	print(cv2.boundingRect(cnt))
	org1 = cv2.rectangle(org1,(x,y),(x+w,y+h),(255,255,0),1)
	if((w>35)&(w<50)&(h>120)&(h<140)):
		crop_imgs.append(copy[y:y+h, x:x+w])

for i in range(len(crop_imgs)):
	cv2.imshow(('Croped '+str(i)),crop_imgs[i])
cv2.imshow('Original',org)
cv2.imshow('Cuted',org1)
cv2.imshow('copy',copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

