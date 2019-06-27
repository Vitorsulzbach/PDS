import numpy as np
import cv2
import PossibleChar
import PossiblePlate
import math


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

listOfPossiblePlates = []

def distanceBetweenChars(firstChar, secondChar):
	intX = abs(firstChar.intCenterX - secondChar.intCenterX)
	intY = abs(firstChar.intCenterY - secondChar.intCenterY)

	return math.sqrt((intX ** 2) + (intY ** 2))

def findListOfMatchingChars(possibleChar, listOfChars):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
	listOfMatchingChars = []                # this will be the return value

	for possibleMatchingChar in listOfChars:                # for each char in big list
		if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches b/c that would end up double including the current char
			continue                                # so do not add to list of matches and jump back to top of for loop
        # end if
                    # compute stuff to see if chars are a match
		fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

		fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

		fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

		fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
		fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
		if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
			fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
			fltChangeInArea < MAX_CHANGE_IN_AREA and
			fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
			fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

			listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars
        # end if
    # end for

	return listOfMatchingChars                  # return result

def angleBetweenChars(firstChar, secondChar):
	fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
	fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

	if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
		fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
	else:
		fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

	fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

	return fltAngleInDeg

def findListOfListsOfMatchingChars(listOfPossibleChars):
            # with this function, we start off with all the possible chars in one big list
            # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
            # note that chars that are not found to be in a group of matches do not need to be considered further
	listOfListsOfMatchingChars = []                  # this will be the return value

	for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
		listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char

		listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars

		if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # if current possible list of matching chars is not long enough to constitute a possible plate
			continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                # to save the list in any way since it did not have enough chars to be a possible plate
        # end if

                                                # if we get here, the current list passed test as a "group" or "cluster" of matching chars
		listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

		listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                # make sure to make a new big list for this since we don't want to change the original big list
		listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

		recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

		for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
			listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
        # end for

		break       # exit for

    # end for

	return listOfListsOfMatchingChars

def extractPlate(imgOriginal, listOfMatchingChars):
	possiblePlate = PossiblePlate.PossiblePlate()           # this will be the return value

	listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the plate
	fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
	fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

	ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculate plate width and height
	intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

	intTotalOfCharHeights = 0

	for matchingChar in listOfMatchingChars:
		intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

	fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

	intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # calculate correction angle of plate region
	fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
	fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
	fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
	fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
	possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
	rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

	height, width, numChannels = imgOriginal.shape      # unpack original image width and height

	imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

	imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

	possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

	return possiblePlate

def getplate(address):
	kNearest = cv2.ml.KNearest_create()
	allContoursWithData = []                # declare empty lists,
	validContoursWithData = []              # we will fill these shortly
	npaClassifications = np.loadtxt("classifications.txt", np.float32) # read in training classifications
	npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)# read in training images
	npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
	kNearest.setDefaultK(1)                                                             # set default K to 1
	kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

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
	contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours
	height1, width1 = imgThresh.shape
	imgContours = np.zeros((height1, width1, 3), np.uint8)
	for i in range(0, len(contours)):                       # for each contour
		possibleChar = PossibleChar.PossibleChar(contours[i])
		if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
		possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
		MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
			intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
			listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars
		# end if
	    # end for

	listOfListsOfMatchingChars = []                  # this will be the return value
	for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
		listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char

		listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars

		if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # if current possible list of matching chars is not long enough to constitute a possible plate
		    continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
		                                        # to save the list in any way since it did not have enough chars to be a possible plate
		# end if

		                                        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
		listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

		listOfPossibleCharsWithCurrentMatchesRemoved = []

		                                        # remove the current list of matching chars from the big list so we don't use those same chars twice,
		                                        # make sure to make a new big list for this since we don't want to change the original big list
		listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

		recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

		for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
			listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
		# end for

		break       # exit for

	# end for
	listOfListsOfMatchingCharsInScene = listOfListsOfMatchingChars
	for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
		possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         # attempt to extract plate

		if possiblePlate.imgPlate is not None:                          # if plate was found
			listOfPossiblePlates.append(possiblePlate)                  # add to list of possible plates
		# end if
	# end for

	print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 with MCLRNF1 image
	return listOfPossiblePlates[0].imgPlate
