import cv2
import os
import numpy as np
from numpy import linalg as LA
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
#plt.style.use('ggplot')

import haarcascade_object_locator

cwd = os.getcwd()

def get_project_matrix():
    ### use orl face dataset to compute a projection matrix
    orl_data_path = 'orl_faces'
    n = 10
    for i in range(1,41):
        sub_folder_str = 's'+ str(i)
        for j in range(1,(n+1)):
            file_name_str = str(j)+'.pgm'
            path = os.path.join(cwd,orl_data_path,sub_folder_str,file_name_str)
            img = cv2.imread(path,-1)
            img = misc.imresize(img,[92,92]) #original orl dataset size is 112*92
            img_flip = img[:, ::-1]
            
            a = img.reshape(1,img.shape[0],img.shape[1])
            a_flip = img_flip.reshape(1,img_flip.shape[0],img_flip.shape[1])
    
            if (i==1) and (j==1): 
                orl_img_set = a
            else:
                orl_img_set = np.concatenate((orl_img_set,a),axis=0)
                
            orl_img_set = np.concatenate((orl_img_set,a_flip),axis=0)
    
    A = np.array(orl_img_set)
    #plt.imshow(A[0],cmap='gray')
    ###
    
    abar = np.mean(A,axis=0) #avg face
    plt.imshow(abar,cmap='gray') 
    
    
    for i in range(A.shape[0]):
        tmp = np.matmul(np.transpose(A[0]-abar),(A[0]-abar))  #(A-Abra)^T * (A-Abar)
        tmp = tmp.reshape(tmp.shape[0],tmp.shape[1],1)
        if i ==0:
            g_tmp = tmp
        else:
            g_tmp = np.concatenate((g_tmp,tmp),axis=2)
    
    G = np.mean(g_tmp,axis=2)
    
    eigenvalue = LA.eig(G)[0]
    eigenvector =LA.eig(G)[1]
    
    cusum =[]
    for i in range(len(eigenvalue)):
        cusum.append(np.sum(eigenvalue[:i])/np.sum(eigenvalue))
    plt.plot(cusum,marker='.')
    
    X = eigenvector
    
    #save model
    np.save('project_matrix.npy',X)
    
    return X



def registered_face_project(X):
    
    train_data_list = os.listdir('registered')
    all_eigen_face = {}
    for i in range(len(train_data_list)):
        imagePath = os.path.join( cwd,'registered',train_data_list[i])
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.imshow(gray,cmap='gray')
        
        gray_resize = misc.imresize(gray,[92,92])
        gg = np.array(gray_resize,dtype = 'float64')
        eigen_face = np.matmul(gg,X)
        
        all_eigen_face.update({train_data_list[i]:eigen_face})
    
    #reconstruct_face = np.matmul(eigen_face,np.transpose(X))
    #plt.imshow(reconstruct_face,cmap='gray')
    
    return all_eigen_face


## testing 
def face_comparison(test_imgPath,X):
    #test_imgPath = 'test.jpg'
    test_img = cv2.imread(test_imgPath)
    
    haarcascade_object_locator.save_flag = False
    faces = haarcascade_object_locator.haarcascade_object_locator(test_imgPath)
    
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(test_gray,cmap='gray')
    
    
    all_eigen_face = registered_face_project(X)
    train_data_list = [key for key in all_eigen_face]
    
    pred_result = []
    for i in range(len(faces)):
        face = faces[i,]
        (x,y,w,h) = face
        single_face = test_gray[(y):(y+h),(x):(x+w)]
        #plt.imshow(single_face)
        single_resize = misc.imresize(single_face,[92,92])
        test_eigen_face = np.matmul(np.array(single_resize,dtype = 'float64'),X)
        
        dd = []
        for key in all_eigen_face:
            d = pc_dist(all_eigen_face[key] , test_eigen_face)
            dd.append(d)
            
        pred_result.append(train_data_list[np.argmin(dd)])
        
        
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = train_data_list[np.argmin(dd)]
        cv2.putText(test_img, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
        
    #plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))  
    test_img = test_img[: , : , : : -1]
    img2 = Image.fromarray(test_img, 'RGB')
    img2.show()
    

    
    return pred_result


def pc_dist(x1,x2):
    dsqrt = (x1-x2)**2
    return np.sum(np.sqrt(np.sum(dsqrt,axis = 0)))



if __name__ == '__main__':
    try:
        X = np.load('project_matrix.npy')
    except:
        X = get_project_matrix()
    
    X = X[:,:20]
    face_comparison('test.jpg',X)
    
    
    
    
    
