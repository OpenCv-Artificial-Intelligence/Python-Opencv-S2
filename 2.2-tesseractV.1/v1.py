import numpy as np
import cv2
import sys
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

#cap = cv2.VideoCapture(0)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#while(True):
# Capture frame-by-frame
# ret, frame = cap.read()

# Our operations on the frame come here
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#print(pytesseract.image_to_string(gray,lang='tha'))
print(pytesseract.image_to_string(Image.open('test-thai.png'), lang='tha'))

# Display the resulting frame
# cv2.imshow('frame',gray)



# When everything done, release the capture
# cap.release()
#cv2.destroyAllWindows()
