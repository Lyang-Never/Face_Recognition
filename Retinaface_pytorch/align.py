import cv2
import os
import numpy as np
from skimage import transform as trans
from PIL import Image

from test_fddb import detect


def read_image(img_path, **kwargs):
  mode = kwargs.get('mode', 'rgb')
  layout = kwargs.get('layout', 'HWC')
  if mode=='gray':
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  else:
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
    if mode=='rgb':
      #print('to rgb')
      img = img[...,::-1]
    if layout=='CHW':
      img = np.transpose(img, (2,0,1))
  return img

def preprocess(img, bbox_list=None, landmark_list=None, **kwargs):
    warped_list = list()
    for bbox,landmark in zip(bbox_list,landmark_list):
        
        if isinstance(img, str):
            img = read_image(img,bbox, **kwargs)
            
        M = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size)>0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size)==1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size)==2
            assert image_size[0]==112
            assert image_size[0]==112 or image_size[1]==96
        if landmark is not None:
            assert len(image_size)==2
            src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
            if image_size[1]==112:
                src[:,0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
            #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        if M is None:
            if bbox is None: #use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if len(image_size)>0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret 
        else: #do align using landmark
            assert len(image_size)==2

            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
            warped_list.append(warped)

    
    return warped_list

def Align(img_path):
    _ret = detect(img_path)
    ret = _ret[0]

    scores = ret[:,4]
    index = list()
    for i in range(len(scores)):
        if scores[i]>0.5:
            index.append(i)
        else: 
            break
            
    bbox = ret[index,0:4]
    points_ = ret[index,5:15]
    points = points_.copy()

    points[index,0:5:1] = points_[index,0::2]
    points[index,5:10:1] = points_[index,1::2] 

    if bbox.shape[0]==0:
      return None
    #bbox = bbox[0,0:4]
    points_list = list()
    bbox_list = []
    for i in range(len(index)):

        point = points[i,:].reshape((2,5)).T
        bbox_ = bbox[i]
        points_list.append(point)
        bbox_list.append(bbox_)
    #points = points.reshape((len(index),2,5)).transpose(0,2,1)
    

    if type(img_path) is not np.ndarray:
        # img = Image.open(img_path)
        # if img.mode == 'L':
        #     img = img.convert('RGB')
        # img_raw = np.array(img)
        face_img = cv2.imread(img_path)
    else:
        face_img = img_path
    
    
    #face_img = cv2.cvtColor(face_img,cv2.COLOR_RGB2BGR)
    #face_img = cv2.imread(img_path)
    nimg_list = preprocess(face_img, bbox_list, points_list, image_size='112,112')
    aligned_list = []
    for _,nimg in enumerate(nimg_list):
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        aligned_list.append(aligned)
    return aligned_list,bbox_list

if __name__ == "__main__":

    path = './features_saveimg'

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            img_root = os.path.join(root, dir)
            images = os.listdir(img_root)
            for image in images:
                path = os.path.join(img_root,image)
                print(path)
                out_list,_ = Align(path)
                for i,out in enumerate(out_list):
                    new_image = np.transpose(out, (1, 2, 0))[:, :, ::-1]
                    out = Image.fromarray(new_image)
                    out = out.resize((112, 112))
                    out = np.asarray(out)

                    if not os.path.exists(os.path.join(root,'_'+dir)):
                        os.mkdir(os.path.join(root,'_'+dir))
                    cv2.imwrite(os.path.join(root,'_'+dir,str(image[0:-4])+'_'+str(i)+'.jpg'),out)
                print('over!')
        # cv2.imshow("out",out)
        # cv2.waitKey()