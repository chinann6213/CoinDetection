"""
coinCounting.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from math import pi

# you are allowed to import other Python packages above
##########################

def findSegment(input):
    
    # using watershed segmentation to identify unique segments in the input image

    D = ndimage.distance_transform_edt(input)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=input)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=input)

    return labels

def findCoinsArea(segmentsFound, oriImg, maskImg, oldCoin=True):
    
    # value of coins
    value = 0 
    for label in np.unique(segmentsFound):
        # if the label is zero, meaning it is 'background'
        # so simply ignore it
        if label == 0:
            continue
     
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(maskImg.shape, dtype="uint8")
        mask[segmentsFound == label] = 255
     
        # detect contours in the mask and get the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
         
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        
        # calculate area based on the radius 
        area = round((pi * r * r), -2)
        if oldCoin == True:
            if area >= 1500 and area < 3201:
                value += 0.05
            elif area >= 3201 and area < 4600:
                value += 0.10 
            elif area >= 4600 and area < 6500:
                value += 0.20
            elif area >= 6500:
                value += 0.50
        elif oldCoin == False:
            if area >= 1500 and area < 2700:
                value += 0.05
            elif area >= 2700 and area < 3200:
                value += 0.10
            elif area >= 3200 and area < 3600:
                value += 0.20
            elif area >= 3600:
                value += 0.50

        # draw circle on the original image for visualizing purpose
        cv2.circle(oriImg, (int(x), int(y)), int(r), (0, 255, 0), 2)
        
        # label the coins 
        if oldCoin == True:
            coinType = "g" # g for gold
        else:
            coinType = "s" # s for silver
        cv2.putText(oriImg, coinType + " #{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    # show the output image
    cv2.imshow("Output", oriImg)
    cv2.waitKey(0)
    return round(value, 2)

def coinCount(coinMat, i):
    # Inputs
    # coinMat: 4-D numpy array of row*col*3*numImages, 
    #          numImage denote number of images in coin set (10 in this case)
    # i: coin set image number (1, 2, ... 10)
    # Output
    # ans: Total value of the coins in the image, in float type
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    image = coinMat[:,:,:,i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel = np.ones((3,3),np.uint8)

    #########################################################################
    # RGB IMAGE PROCESSING
    
    edge = cv2.Canny(image, 100, 200)
  
    edge = cv2.dilate(edge, kernel, iterations = 2)
    edge = cv2.erode(edge, kernel, iterations = 2)

    cnts = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(edge, cnts, -1, (255,255,255), 4)

    edge = cv2.dilate(edge, kernel, iterations = 2)
    edge = cv2.erode(edge, kernel, iterations = 2)
  
    floodFillEdge = edge.copy()
    h, w = edge.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(floodFillEdge, mask, (0,0), 255);
    floodFillEdgeInv = cv2.bitwise_not(floodFillEdge)
    floodedEdge = edge | floodFillEdgeInv

    #########################################################################
    # HSV IMAGE PROCESSING
    
    hsv = cv2.pyrMeanShiftFiltering(image, 2, 40)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
  
    S = cv2.threshold(S, 0, 255, cv2.THRESH_OTSU)[1]
    if S[0, int(S.shape[1]/2)] == 255:
        S = (255-S) # invert image color
    
    floodFillS = S.copy()
    h, w = S.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(floodFillS, mask, (0,0), 255);
    floodFillSInv = cv2.bitwise_not(floodFillS)
    S = S | floodFillSInv
  
    S = cv2.erode(S, kernel, iterations = 8)
    S = cv2.dilate(S, kernel, iterations = 8)
 
    #########################################################################
    # IMAGE SUBTRACTION BETWEEN S(MOST WITH GOLD COIN) AND floodedEdge(ALL COIN)

    subtracted = S - floodedEdge
    subtracted = np.sqrt((subtracted ** 2))
    subtracted = cv2.erode(subtracted, kernel, iterations = 4)
    subtracted = cv2.dilate(subtracted, kernel, iterations = 4)
  
    #########################################################################
    # DISPLAYING

    floodedEdgeDisplay = cv2.resize(floodedEdge, (400, 264))
    floodedEdgeDisplay = cv2.copyMakeBorder(floodedEdgeDisplay,1,1,1,1,cv2.BORDER_CONSTANT,value=255)

    SDisplay = cv2.resize(S, (400, 264))
    SDisplay = cv2.copyMakeBorder(SDisplay,1,1,1,1,cv2.BORDER_CONSTANT,value=255)

    SubtractedDisplay = cv2.resize(subtracted, (400, 264))
    SubtractedDisplay = cv2.copyMakeBorder(SubtractedDisplay,1,1,1,1,cv2.BORDER_CONSTANT,value=255)

    stack = np.hstack((floodedEdgeDisplay, SDisplay, SubtractedDisplay))
    cv2.imshow("All Coins -- Saturation Layer -- Subtracted", stack)

    
    #########################################################################
    # COINS DETECTION USING WATERSHED AND COINS VALUE CALCULATION

    watershedInput = S.copy()
    SSegments = findSegment(watershedInput) 
    watershedInput = floodedEdge.copy()
    floodedEdgeSegments = findSegment(watershedInput)

    print("[INFO]" + "Image " + str(i+1))

    if len(np.unique(SSegments)) - 1 == len(np.unique(floodedEdgeSegments)) - 1:
        watershedInput = floodedEdge.copy()
        floodedEdgeSegments = findSegment(watershedInput)
        print("{} unique segments found in image with all coins (floodedEdge)\n".format(len(np.unique(floodedEdgeSegments)) - 1))
        ans =  findCoinsArea(floodedEdgeSegments, image, edge, oldCoin = True)
    else:
        watershedInput = S.copy()
        SSegments = findSegment(watershedInput)
        print("{} unique segments found in Saturation Layer\n".format(len(np.unique(SSegments)) - 1))
        coinsDetectedS = findCoinsArea(SSegments, image, edge, oldCoin = False)
       
        watershedInput = subtracted.copy()
        subtractedSegments = findSegment(watershedInput)
        print("{} unique segments found in subtracted image\n".format(len(np.unique(subtractedSegments)) - 1))
        coinsDetectedSub = findCoinsArea(subtractedSegments, image, edge, oldCoin = True)
        ans = coinsDetectedS + coinsDetectedSub

    # END OF YOUR CODE
    #########################################################################
    

    return round(ans, 2)
    