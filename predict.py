# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

import cv2 as cv
import numpy as np
from keras.models import load_model

label = np.array(["ALPHA","BETA","CHI","DELTA","EPSILON","ETA","GAMMA","IOTA","KAPPA","LAMDA","MU","NU","OMEGA","OMICRON","PHI","PI","PSI","RHO","SIGMA","TAU","THETA","UPSILON","XI","ZETA"])

def inverse_t(pred):
  for i in range(len(pred[0])):
    if pred[0][i] == 1:
      return label[i]

def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration
	model = load_model('my_model_1.h5')
	codes=[]
	for filename in filenames:
		image = cv.imread(filename)
       
		hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
		lower_gray = np.array([0,0,150])
		upper_gray = np.array([255,255,255])
       
		mask = cv.inRange(hsv, lower_gray, upper_gray)
		imCopy = mask.copy()
		imCopy=~imCopy
       
		contours,hierarchy= cv.findContours(imCopy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		letter_image_regions = []

        # Now we can loop through each of the contours and extract the letter
		for contour in contours:
            # Get the rectangle that contains the contour
			(x, y, w, h) = cv.boundingRect(contour)
            
            # checking if any counter is too wide
            # if countour is too wide then there could be two letters joined together or are very close to each other
            # if w / h > 1.5:
            #     # Split it in half into two letter regions
            #     half_width = int(w / 2)
            #     letter_image_regions.append((x, y, half_width, h))
            #     letter_image_regions.append((x + half_width, y, half_width, h))
            # else:
			letter_image_regions.append((x, y, w, h))
                    

        # Sort the detected letter images based on the x coordinate to make sure
        # we get them from left-to-right so that we match the right image with the right letter
		letter_image_regions = sorted(letter_image_regions, key=lambda x: x[2]*x[3], reverse=True)  
		letter_image_regions = letter_image_regions[0:3]
		letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Create an output image and a list to hold our predicted letters
		output = cv.merge([mask] * 3)
		predictions = []
            
        # Creating an empty list for storing predicted letters
		predictions = []
        # Save out each letter as a single image
		for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
			x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
			letter_image = mask[y - 2:y + h + 2, x - 2:x + w + 2]

			letter_image = cv.resize(letter_image, (30,30))
                
            # Turn the single image into a 4d list of images
			letter_image = np.expand_dims(letter_image, axis=2)
			letter_image = np.expand_dims(letter_image, axis=0)

            # making prediction
			pred = model.predict(letter_image)
                
            # Convert the one-hot-encoded prediction back to a normal greek letter
            # print(pred)
			letter = inverse_t(pred)
			predictions.append(letter)
        # Print the captcha's text
        # predictions = handleMultipleXi(predictions)
        # print(predictions.dtype())
		captcha_text = ",".join(predictions)
		codes.append(captcha_text)
	return codes