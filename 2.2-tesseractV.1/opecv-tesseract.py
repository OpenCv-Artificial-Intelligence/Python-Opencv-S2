import cv2
import sys
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

if __name__ == '__main__':

     
    # If you don't have tesseract executable in your PATH, include the following:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
    

    # Simple image to string
    image = cv2.imread('3.png',0)
    #image2 = cv2.imread('test.png')

    #แบบใช่ opencv คือการดึงรูปมาตรงๆ ผลลัพตรง
    print(pytesseract.image_to_string(image,lang='tha'))
    #print(pytesseract.image_to_string(image2,lang='eng'))

    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #แบบไม่ใช่ opencv คือการดึงรูปมาตรงๆ ผลลัพค่อนค้างเพี้ยน 
    #print(pytesseract.image_to_string(Image.open('test-thai.png'),lang='tha'))

