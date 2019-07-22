import cv2
import numpy as np
import operator
import os


class ContourWithData():

    npaContour = None           
    boundingRect = None         
    intRectX = 0                
    intRectY = 0                
    intRectWidth = 0            
    intRectHeight = 0           
    fltArea = 0.0               

    def calculateRectTopLeftPointAndWidthAndHeight(self):               
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            
        if self.fltArea < 100: return False        
        return True


def main():
    allContoursWithData = []                
    validContoursWithData = []              

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return


    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       

    kNearest = cv2.ml.KNearest_create()                   

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    img_TestingNumbers = cv2.imread("6.png")          

    if img_TestingNumbers is None:                           
        print ("error: image not read from file \n\n")      
        os.system("pause")                                  
        return                                              


    img_Gray = cv2.cvtColor(img_TestingNumbers, cv2.COLOR_BGR2GRAY)       
    img_Blurred = cv2.GaussianBlur(img_Gray, (5,5), 0)                    

                                                        
    img_Thresh = cv2.adaptiveThreshold(img_Blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY_INV, 11, 2)                                    

    img_ThreshCopy = img_Thresh.copy()        

    img_Contours, npaContours, npaHierarchy = cv2.findContours(img_ThreshCopy,             
                                                 cv2.RETR_EXTERNAL,         
                                                 cv2.CHAIN_APPROX_SIMPLE)   

    #cv2.imshow("img_ThreshCopy", img_ThreshCopy)

    for npaContour in npaContours:                                                      
        if cv2.contourArea(npaContour)>50:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
            cv2.drawContours(img_ThreshCopy, [npaContour], 0, (255,255,0),2)
            npaContour_len = cv2.arcLength(npaContour, True)
            npaContour = cv2.approxPolyDP(npaContour, 0.02*npaContour_len, True)
            Area = cv2.contourArea(npaContour)
            if 90 < intH < 130:                             
            #if 120 < intH < 160:
                       contourWithData = ContourWithData()                                             
                       contourWithData.npaContour = npaContour                                         
                       contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     
                       contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    
                       contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           
                       allContoursWithData.append(contourWithData)                                     


    for contourWithData in allContoursWithData:                 
        if contourWithData.checkIfContourIsValid():             
            validContoursWithData.append(contourWithData)       


    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         

    strFinalString = ""         

    for contourWithData in validContoursWithData:            
                                                
        cv2.rectangle(img_TestingNumbers, (contourWithData.intRectX, contourWithData.intRectY), (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight), (255, 255, 0), 2)                        

        img_ROI = img_Thresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight, contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        img_ROIResized = cv2.resize(img_ROI, (20, 30))             

        npaROIResized = img_ROIResized.reshape((1, 20 * 30))      

        npaROIResized = np.float32(npaROIResized)       

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 3)
        
        if int(npaResults[0][0])<160:
            strCurrentChar = str(chr(int(npaResults[0][0])))                                             
        else:
            strCurrentChar = str(chr(int(npaResults[0][0])+3424))
            
        strFinalString = strFinalString + strCurrentChar            

    print ("\n" + strFinalString + "\n")  

    cv2.imshow("img_TestingNumbers", img_TestingNumbers)      
    cv2.waitKey(0)                                          

    cv2.destroyAllWindows()             

    return


if __name__ == "__main__":
    main()










