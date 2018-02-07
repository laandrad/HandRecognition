# Webcam Hand Tracking

We are developing a deep learning algorithm for traking a hand on a webcam image. The goal is to use this as an input interface for games and learning applications on mobile devices.

Flowchart
1. TDT_function2: takes in raw images, resizes them and outputs train, dev, and test sets
2. Get_data_model2: takes in processed images and outputs a 4D numpy array
3. model2: takes in 4D array and trains a CNN model
4. CNN_to_New_Features: takes in 4D array and outputs a 128-m array
5. SVM_CNN_Features: takes in 128-m array and trains an SVM model

## Useful links to hand images data sets
- https://www.mutah.edu.jo/biometrix/hand-images-databases.html  
- https://sites.google.com/view/11khands  
- http://biometrics.idealtest.org/dbDetailForUser.do?id=5. 
- http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm#gesture

## Useful links to object classification with opencv and python
- https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
- https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
- https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

## Link to our hands database
https://www.dropbox.com/sh/o3yv866isxvj7qt/AAC3ipCzJde86d53QuzlZEEia?dl=0
