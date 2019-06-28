import numpy as np
import cv2
import PossibleChar
import PossiblePlate
import math
import pytesseract

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MIN_PIXEL_AREA = 80
MAX_ASPECT_RATIO = 1.0
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
MIN_CONTOUR_AREA = 100

def distanceBetweenChars(firstChar, secondChar):
	intX = abs(firstChar.intCenterX - secondChar.intCenterX)
	intY = abs(firstChar.intCenterY - secondChar.intCenterY)
	return math.sqrt((intX ** 2) + (intY ** 2))

def findListOfMatchingChars(possibleChar, listOfChars):
	listOfMatchingChars = []
	for possibleMatchingChar in listOfChars:
		if possibleMatchingChar == possibleChar:
			continue
		fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
		fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
		fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
		fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
		fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
		if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
			fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
			fltChangeInArea < MAX_CHANGE_IN_AREA and
			fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
			fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
			listOfMatchingChars.append(possibleMatchingChar)
	return listOfMatchingChars

def angleBetweenChars(firstChar, secondChar):
	fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
	fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
	if fltAdj != 0.0:
		fltAngleInRad = math.atan(fltOpp / fltAdj)
	else:
		fltAngleInRad = 1.5708
	fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)
	return fltAngleInDeg

def findListOfListsOfMatchingChars(listOfPossibleChars):
	listOfListsOfMatchingChars = []
	for possibleChar in listOfPossibleChars:
		listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
		listOfMatchingChars.append(possibleChar)

		if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
			continue
		listOfListsOfMatchingChars.append(listOfMatchingChars)
		listOfPossibleCharsWithCurrentMatchesRemoved = []
		listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

		recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)
		for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
			listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
		break
	return listOfListsOfMatchingChars

def extractPlate(imgOriginal, listOfMatchingChars):
	possiblePlate = PossiblePlate.PossiblePlate()
	listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)
	fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
	fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
	ptPlateCenter = fltPlateCenterX, fltPlateCenterY
	intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
	intTotalOfCharHeights = 0
	for matchingChar in listOfMatchingChars:
		intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
	fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
	intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
	fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
	fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
	fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
	fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
	possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
	rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
	height, width, numChannels = imgOriginal.shape
	imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))
	imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
	possiblePlate.imgPlate = imgCropped
	return possiblePlate

def getplate(address):
	listOfPossiblePlates = []
	imgOriginalScene  = cv2.imread(address)
	ADAPTIVE_THRESH_WEIGHT = 9
	ADAPTIVE_THRESH_BLOCK_SIZE = 19
	GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
	height, width, numChannels = imgOriginalScene.shape
	imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
	imgThreshScene = np.zeros((height, width, 1), np.uint8)
	imgContours = np.zeros((height, width, 3), np.uint8)
	imgHSV = np.zeros((height, width, 3), np.uint8)
	imgHSV = cv2.cvtColor(imgOriginalScene, cv2.COLOR_BGR2HSV)
	imgHue, imgSaturation, imgGrayscale = cv2.split(imgHSV)
	height1, width1 = imgGrayscale.shape
	imgTopHat = np.zeros((height1, width1, 1), np.uint8)
	imgBlackHat = np.zeros((height1, width1, 1), np.uint8)
	structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
	imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
	imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
	imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
	imgBlurred = np.zeros((height1, width1, 1), np.uint8)
	imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
	imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
	imgGrayscaleScene = imgGrayscale
	imgThreshScene = imgThresh
	listOfPossibleChars = []
	intCountOfPossibleChars = 0
	imgThreshCopy = imgThresh.copy()
	contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	height1, width1 = imgThresh.shape
	imgContours = np.zeros((height1, width1, 3), np.uint8)
	for i in range(0, len(contours)):
		possibleChar = PossibleChar.PossibleChar(contours[i])
		if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
		possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
		MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
			intCountOfPossibleChars = intCountOfPossibleChars + 1
			listOfPossibleChars.append(possibleChar)
	listOfListsOfMatchingChars = []
	for possibleChar in listOfPossibleChars:
		listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
		listOfMatchingChars.append(possibleChar)
		if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:

		    continue
		listOfListsOfMatchingChars.append(listOfMatchingChars)
		listOfPossibleCharsWithCurrentMatchesRemoved = []
		listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

		recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)
		for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
			listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
		break
	listOfListsOfMatchingCharsInScene = listOfListsOfMatchingChars
	for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
		possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

		if possiblePlate.imgPlate is not None:
			listOfPossiblePlates.append(possiblePlate)

	#Seleção das possíveis placas:
	#J=0
	lis = []
	for l in listOfPossiblePlates:
		i=0
		img = cv2.cvtColor(l.imgPlate, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(512,200))
		img = abs(255-img)
		kernel = np.ones((5,5), np.uint8)
		img1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
		img2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
		img = cv2.add(img,img1)
		img = cv2.absdiff(img,img2)
		img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,0)
		img = cv2.GaussianBlur(img,(3,3),0)
		a,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		kernel = np.ones((2,2), np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			if((w>35)&(w<70)&(h>100)&(h<160)&(y>3)&(x>3)):
				copy=img[y-4:y+h+8, x-4:x+w+8]
				c = pytesseract.image_to_string(copy,config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 0')
				if((c=='A')|(c=='B')|(c=='C')|(c=='D')|(c=='E')|(c=='F')|(c=='G')|(c=='J')|(c=='K')|(c=='L')|(c=='M')|(c=='N')|(c=='O')|(c=='P')|(c=='Q')|(c=='R')|(c=='S')|(c=='T')|(c=='U')|(c=='V')|(c=='W')|(c=='X')|(c=='Y')|(c=='Z')|(c=='0')|(c=='2')|(c=='3')|(c=='4')|(c=='5')|(c=='6')|(c=='7')|(c=='8')|(c=='9')):
					#print(c)
					i=i+1
		if(i<8):
			#cv2.imshow(('debug'+str(J)),img)
			#print('added: '+str(i)+','+str(J))
			lis.append((l.imgPlate,i))
		#J=J+1

	lis.sort(key = lambda x: int(x[1]),reverse=True)
	

	#lis[0][0] é a placa escolhida
	#print(lis[0][1])
	#cv2.imshow('chosen',lis[0][0])
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return lis[0][0]



