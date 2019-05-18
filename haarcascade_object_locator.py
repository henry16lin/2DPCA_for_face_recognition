import cv2
import os
from PIL import Image

cascName = 'haarcascade_frontalface_default.xml'
save_path = 'registered'
save_flag = True

def haarcascade_object_locator(imagePath,video_flag=0): 
    # Create the haar cascade
    cascPath = os.path.join(os.getcwd(),'haarcascade',cascName)
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    # Read the image
    if video_flag:
        image = imagePath
    else:
        image = cv2.imread(imagePath)
    
    image_resize = image
    gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    try:
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        print("Found {0} objects!".format(len(faces)))
    except:
        print('**\n object locate fail...check cascade.xml file...\n**')   
    
    if save_flag:
        ori_img = image_resize
        for i in range(len(faces)):
            face = faces[i,]
            (x,y,w,h) = face
            single_face = ori_img[(y):(y+h),(x):(x+w),:]
            #plt.imshow(single_face)
            if save_flag:
                face_name = str(i)+'.jpg'
                full_save_path = os.path.join(os.getcwd(),save_path,face_name)
                cv2.imwrite(full_save_path,single_face)
            
            cv2.rectangle(image_resize, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = 'object'
            cv2.putText(image_resize, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    

        image_resize = image_resize[: , : , : : -1]
        img2 = Image.fromarray(image_resize, 'RGB')
        img2.show()
    
    return faces

#haarcascade_object_locator('5566.jpg')


