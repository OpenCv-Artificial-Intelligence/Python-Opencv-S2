import sys
import numpy as np
import cv2
import os


def main():
    img_TrainingNumbers = cv2.imread("training1.png")            

    if img_TrainingNumbers is None:                          
        print ("error: image not read from file \n\n")      
        os.system("pause")                                  
        return                                              
    

    img_Gray = cv2.cvtColor(img_TrainingNumbers, cv2.COLOR_BGR2GRAY)          
    img_Blurred = cv2.GaussianBlur(img_Gray, (5,5), 0)                        

                                                        
    img_Thresh = cv2.adaptiveThreshold(img_Blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)                                    

    cv2.imshow("imgThresh", img_Thresh)      

    img_ThreshCopy = img_Thresh.copy()        

    img_Contours, npaContours, npaHierarchy = cv2.findContours(img_ThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)           

                                                               
    npaFlattenedImages =  np.empty((0, 20 * 30))

    intClassifications = []         

    # intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
    #                  ord('ก'), ord('ข'), ord('ฃ'), ord('ค'), ord('ฅ'), ord('ฆ'), ord('ง'), ord('จ'), ord('ฉ'), ord('ช'),
    #                  ord('ซ'), ord('ฌ'), ord('ญ'), ord('ฎ'), ord('ฏ'), ord('ฐ'), ord('ฑ'), ord('ฒ'), ord('ณ'), ord('ด'),
    #                  ord('ต'), ord('ถ'), ord('ท'), ord('ธ'), ord('น'), ord('บ'), ord('ป'), ord('ผ'), ord('ฝ'), ord('พ'), 
    #                  ord('ฟ'), ord('ภ'), ord('ม'), ord('ย'), ord('ร'), ord('ล'), ord('ว'), ord('ศ'), ord('ษ'), ord('ส'), 
    #                  ord('ห'), ord('ฬ'), ord('อ'), ord('ฮ')]

    intValidChars = [k for k in range(48,207)]

    for npaContour in npaContours:                          
        if cv2.contourArea(npaContour) > 100:          
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         
                                                
            cv2.rectangle(img_TrainingNumbers, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)                            

            img_ROI = img_Thresh[intY:intY+intH, intX:intX+intW]                                  
            img_ROIResized = cv2.resize(img_ROI, (20, 30))     

            cv2.imshow("imgROI", img_ROI)                    
            cv2.imshow("imgROIResized", img_ROIResized)      
            cv2.imshow("training_numbers.png", img_TrainingNumbers)      

            intChar = cv2.waitKey(0)                     

            if intChar == 27:                   
                sys.exit()                      
            elif intChar in intValidChars:      

                intClassifications.append(intChar)                                                

                npaFlattenedImage = img_ROIResized.reshape((1, 20 * 30))  
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    


    fltClassifications = np.array(intClassifications, np.float32)                   

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           
    np.savetxt("flattened_images.txt", npaFlattenedImages)          

    cv2.destroyAllWindows()             

    return


if __name__ == "__main__":
    main()







