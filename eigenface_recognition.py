import argparse
import os
import cv2
import numpy as np
from PIL import Image
import haarcascade_object_locator

cwd = os.getcwd()
def load_registed_img(ref_face_folder):
    class_folder_list =  os.listdir(os.path.join(cwd,ref_face_folder))
    X,y,ind = [],[],[]
    i = 0
    for c in class_folder_list:
        single_class_path = os.path.join(cwd,ref_face_folder,c)
        for img in os.listdir(single_class_path):
            image = cv2.imread(os.path.join(single_class_path,img))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_resize = cv2.resize(gray,(200,200))
            
            X.append(np.asarray(gray_resize,dtype=np.uint8))
            y.append(c)
            ind.append(i)
        i+=1
        
    return X,y,ind


def eigenspace_tranformer():
    return cv2.face.FisherFaceRecognizer_create(threshold=1000.0)


## prediction 
def face_comparison(test_imgPath,video_flag):
    if not video_flag:
        test_img = cv2.imread(test_imgPath)
    else:
        test_img = test_imgPath
    
    haarcascade_object_locator.save_flag = False
    faces = haarcascade_object_locator.haarcascade_object_locator(test_imgPath,video_flag)
    
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(test_gray,cmap='gray')
 
    
    for i in range(len(faces)):
        face = faces[i,]
        (x,y,w,h) = face
        single_face = test_gray[(y):(y+h),(x):(x+w)]
        
        single_resize = cv2.resize(single_face,(200,200))
        pred_result = model.predict(single_resize)
        pred,confidence = pred_result[0],pred_result[1]
        
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if pred!=-1:
            cat_ind = int(np.where(np.array(ind)==pred)[0][0])
            
            text = cat[cat_ind] +', ' + str(np.round(confidence,3))
            cv2.putText(test_img, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        else:
            text = 'unknown'
            cv2.putText(test_img, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
        
    if not video_flag:    
        #plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))  
        test_img = test_img[: , : , : : -1]
        img2 = Image.fromarray(test_img, 'RGB')
        img2.show()

    return test_img

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
            '--video', default=False, action="store_true",
            help='if not given, it will use image mode'
        )
    parser.add_argument(
            '--ref_face_folder', type=str, default='registered',
            help='folder where place the face you want to compare'
        )
    parser.add_argument(
            '--input_path', type=str,
            help='image or video path that you want to predict'
        )
    parser.add_argument(
            '--output_path', type=str,
            help='where to save the result'
        )
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    ref_face_folder = args.ref_face_folder
    video_flag = args.video
    
    
    image,cat,ind=load_registed_img(ref_face_folder)
    model = eigenspace_tranformer()
    model.train(np.asarray(image),np.array(ind))
    
    
    if not video_flag:
        test_img = face_comparison(input_path,video_flag)
    
    else:
        '''
        ref_face_folder = 'test'
        video_flag = 1
        input_path = r'video_test2.mp4'
        output_path = os.path.join(cwd,'video_test_result2.mp4')
        '''
        
        cap = cv2.VideoCapture(input_path)
        
        video_FourCC    = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps       = cap.get(cv2.CAP_PROP_FPS)
        video_size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        
        while(True):
            ret, frame = cap.read()
            try:
                image = face_comparison(frame,video_flag)
                result = np.asarray(image)
                cv2.imshow('frame', result)
                out.write(result)
            except:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    

