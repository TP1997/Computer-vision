import numpy as np 
#from cv2 import cv2
import cv2
import os

def CascadeDetection(imageFile, cascadeFile1, cascadeFile2):
    
    # Load the image
    image = cv2.imread(imageFile)
    
    # Create a copy of the image
    imageCopy = image.copy()


    # Convert the image from BGR to Grayscale
    # Load the cascades using cv2.CascadeClassifier
    
    ##-your-code-starts-here-##
    imageCopy_g = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)
    haarCascade1 = cv2.CascadeClassifier(cascadeFile1)
    haarCascade2 = cv2.CascadeClassifier(cascadeFile2)
    #-your-code-ends-here-##

    # Perform multi-scale face detection for the gray image using detectMultiScale
    
    #-your-code-starts-here-##
    detectedObjects = haarCascade1.detectMultiScale(imageCopy_g)
    #-your-code-ends-here-##
    
    #  Draw bounding boxes
    for bbox in detectedObjects:
        # Draw the bounding box with cv2.rectangle
        # Save cropped image to crop-variable based on bounding box coordinates 
        #-your-code-starts-here-##
        x=bbox[0];y=bbox[1];w=bbox[2];h=bbox[3]
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        crop = imageCopy_g[y:y+h, x:x+w] # replace me
        #-your-code-ends-here-##

        # Perform multi-scale smile detection for cropped image using detectMultiScale
        cropDetectedObjects = haarCascade2.detectMultiScale(crop) # replace me
        for bbox2 in cropDetectedObjects:
            # Draw the bounding box with cv2.rectangle

            #-your-code-starts-here-##
            x2=bbox2[0];y2=bbox2[1];w2=bbox2[2];h2=bbox2[3]
            image = cv2.rectangle(image, (x+x2,y+y2), (x+x2+w2,y+y2+h2), (0,255,255), thickness=2)
            #-your-code-ends-here-##
            

    # Display the output
    cv2.imshow("Face and smile detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the bounding boxes
    return detectedObjects

os.chdir('/home/tuomas/Python/DATA.ML.300/Bonus/Exercise Bonus/Task 2/')
smileDetection = CascadeDetection("fallon.jpeg",
                                  "cascades/haarcascade_frontalface_default.xml",
                                  "cascades/haarcascade_smile.xml")
