import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2


def get_affine_transform(size1, size2): 
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    scale1 = size2[0]*1.0/size1[0]
    scale2 = size2[1]*1.0/size1[1]
    scale = min(scale1,scale2)
    # Center to Center
    src[0, :] = [size1[0]/2.0 , size1[1]/2.0]
    dst[0, :] = [size2[0]/2.0 , size2[1]/2.0]

    # Left Center to Left Center Boarder 
    src[1, :] = [0.0 , size1[1]/2.0]
    dst[1, :] = [size2[0]/2.0 - scale*size1[0]/2.0 , size2[1]/2.0]

    # Top Center to Top Center Boader
    src[2, :] = [ size1[0]/2.0, 0.0]
    dst[2, :] = [ size2[0]/2.0 , size2[1]/2.0 - scale*size1[1]/2.0 ]
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def  hsv_transform(img, hue, sat, expo):
    
    dsat = rand_scale(sat)
    dexp = rand_scale(expo)
    
    dhue = np.random.uniform(low=-hue, high=hue)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    
    return img
    


def preprocess_all(img, hue, sat, expo):
    
    w = img.shape[1]
    h = img.shape[0]
    jitter = 0.3
    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw)) / (h + np.random.uniform(low=-dh, high=dh))
        new_w = w + np.random.uniform(low=-dw, high=dw)
        new_h = h + np.random.uniform(low=-dh, high=dh)
        
        img = cv2.resize(img, (int(new_w), int(new_h)))
    #   hue       颜色（hsv颜色空间）数据增强参数：色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值
    #    saturation 颜色（hsv颜色空间）数据增强参数：色彩饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值
    #      exposure  颜色（hsv颜色空间）数据增强参数：明度（色彩明亮程度，0~1）缩放最大值，实际为范围内的随机值
    img = hsv_transform(img, 0.1, 1.5, 1.5)
    
    return img#, ratio_w, ratio_h
    
    
        
    

class COCODataset_Person(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir='../wheat_data/train/', data_dir_labels='../../../../../my_task/data/coco_person_label/',json_file='annotations.json',
                 name='train', img_size=416,augmentation=None, min_size=1, debug=False):
        
        super(COCODataset_Person, self).__init__()
        
        img_label = []
        list_img = []
        img_label_txt_name = []
        self.model_type = model_type
        
        all_img_dir = os.listdir(data_dir)
        #data_dir_labels = "../../../../my_task/data/voc_person_data/"
        #data_dir_labels = "../../../../../my_task/data/coco_person_label/"
        
        
        for tmp_img_name in all_img_dir:
            if ".jpg" in tmp_img_name:
                txt_name = data_dir_labels + tmp_img_name + ".txt" #.split(".")[0]
                #txt_name = data_dir + "labels/" + tmp_img_name.split(".")[0] + ".txt"
                if os.path.exists(txt_name):
                    list_img.append(data_dir + tmp_img_name)
                    img_label_txt_name.append(txt_name)
                      
        self.img_and_label = list_img
        self.img_label_lst = img_label_txt_name
        
        self.nF = len(list_img)
        print("Total {} imgs found.".format(self.nF))
        
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.augmentation = augmentation


    def __len__(self):
        return  self.nF

    def __getitem__(self, file_index):
   
        img_dir = self.img_and_label[file_index]
        #print("img_dir is ", img_dir )
        img_label_txt = self.img_label_lst[file_index]
        #print("img_label_txt is ", img_label_txt )
        img_file = img_dir
        img = cv2.imread(img_file, 1) 
        #cv2.imwrite("debug_org"+".jpg", img)
        #print("img_label_txt is ",img_label_txt )
        
        assert img is not None
        # 将图像映射为41*416
        ##  -- 计算 变换矩阵， 一个用于变换图片，一个用于变换坐标
        if self.augmentation == True:# is not None:
            #print("augmentation is True")
            img = preprocess_all(img, 0.1, 1.5, 1.5)
        
        height = img.shape[0]
        width = img.shape[1]
        
        #print(height,width)
        
        trans = get_affine_transform((width, height),(self.img_size, self.img_size))
        img = cv2.warpAffine(img, trans, (self.img_size, self.img_size))
        img_show = img.copy()
        img= img.astype(np.float32)
        img = np.transpose(img / 255., (2, 0, 1))
      
        labels = []
        #f_l = open()
        
        #try:
        #padded_labels = np.zeros((1,13,13, 5), np.float32)
        padded_labels = np.zeros((50, 5), np.float32)   #max save 50 rect
        
        padded_labels_l  = []
        
        if True:
            txt = open(img_label_txt,'r')
            lines = txt.readlines()
            for i in range(0,len(lines)):
                j = lines[i].strip()

                #category_id = int(j.split(' ')[0])
                category_id = 0
                x=float(j.split(' ')[0])*width
                y=float(j.split(' ')[1])*height
                w=float(j.split(' ')[2])*width
                h=float(j.split(' ')[3])*height
                cls_id = int(j.split(' ')[4])
                
                x1,y1,x2,y2 = x - w/2 , y - h/ 2 , x + w / 2, y + h /2
                
                x1 = np.clip(x1 , 0 , width - 1)
                y1 = np.clip(y1 , 0 , height - 1)
                x2 = np.clip(x2 , 0 , width - 1)
                y2 = np.clip(y2 , 0 , height - 1)
                
                P1 = [[x1],[y1],[1.0]]
                P2 = [[x2],[y2],[1.0]]
                
                P1_new = np.matmul(trans, P1)
                P2_new = np.matmul(trans, P2)
                
                x1 = P1_new[0][0]
                y1 = P1_new[1][0]
                
                x2 = P2_new[0][0]
                y2 = P2_new[1][0]
                
                #cv2.rectangle(img_show,(int(x1),int(y1)),(int(x2),int(y2)),(0, 255, 0), 2)
                
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                
                
                #xc_grid = int(xc/(self.img_size*1.0)*13)
                #yc_grid = int(yc/(self.img_size*1.0)*13)
                #print("xc_grid ", xc_grid, yc_grid )
                
                #xc_grid_delta = xc*1.00/(self.img_size*1.00)#*13.00 - xc_grid
                #yc_grid_delta = yc*1.00/(self.img_size*1.00)#*13.00 - yc_grid
                
                #w_norm = (x2 - x1)*1.0 / self.img_size
                #h_norm = (y2 - y1)*1.0 / self.img_size
                #print("xc_grid_delta", yc_grid_delta, xc_grid_delta, xc_grid, yc_grid)
                
                #padded_labels[0] = [xc_grid_delta, yc_grid_delta, w_norm, h_norm]
                
                padded_labels[i, 0] = xc/self.img_size
                padded_labels[i, 1] = yc/self.img_size
                padded_labels[i, 2] =  (x2 - x1)/self.img_size
                padded_labels[i, 3] = (y2 - y1)/self.img_size
                padded_labels[i, 4] = cls_id
                
                #padded_labels_l.append([ xc/self.img_size,  yc/self.img_size, (x2 - x1)/self.img_size,  (y2 - y1)/self.img_size])
                
                #padded_labels[0, yc_grid, xc_grid, 0] = 1.0
                #padded_labels[0, yc_grid, xc_grid, 1] = xc / self.img_size #xc_grid_delta
                #padded_labels[0, yc_grid, xc_grid, 2] = yc / self.img_size #yc_grid_delta
               # padded_labels[0, yc_grid, xc_grid, 3] = (x2 - x1) / self.img_size # width
                #padded_labels[0, yc_grid, xc_grid, 4] = (y2 - y1) / self.img_size # height
                

#         except:
#             labels = []
        #print("padded_labels is ", padded_labels)
        info_img = []
        id_ = ""
        #cv2.imwrite("debug_"+".jpg", img_show)
        padded_labels_th = torch.from_numpy(padded_labels) #np.array(padded_labels_l))
        return img, padded_labels_th, info_img, id_
    
    
#dataset = COCODataset_Person(model_type='YOLO', data_dir='../../../../../my_task/data/coco_person/coco_2017_person_yolo_format/train/images/', img_size=416, augmentation=True)#
#dataset =COCODataset_Person(model_type='YOLO',data_dir='../../../../../my_task/data/coco_org/train2017/',#coco_person/coco_2017_person_yolo_format/train/images/',
#                 data_dir_labels='../../../../../my_task/data/coco_one_label/', 
#                  img_size = 416,
#                  augmentation = True)
#dataset.__getitem__(15)

