# 2DPCA_for_face_recognition
compare test face to registered faces for face recognition

`2DPCA_face_recognition` compute project matrix by orl face dataset:  
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html  

## NOTE
`haarcascade_object_locator.py`: help you cut face from original image and save to folder registered  
`2DPCA_face_recognition.py`:   
1. create project matrix if there is doesn't exist  
2. load registered face and test image   
3. compute distance between face in the test image and each registered face and do face recognition  

`eigenface_recognition.py`: similar to `2DPCA_face_recognition.py` but projection transformer come from cv2 which is easier to use...

