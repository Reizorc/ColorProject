import cv2
from LBP import LocalBinaryPatterns

# Return the LBP histogramme of an RGB image I
# np = number of points used for lbp
# r = radius of the lbp circle
def texture(I,np = 24,r = 8):

	desc = LocalBinaryPatterns(np,r)
	gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	return hist



