import numpy as np
import os
import sys


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.autograd import Variable
import darknet
import torch.optim as optim
import cv2
import utils

from input import coco_dataset_p
batch_size = 2
torch.cuda.set_device(2)
device = torch.device("cuda:2")

#anchors = [ [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],[7.88282, 3.52778], [9.77052, 9.16828] ]
anchors = [[10,13],  [16,30],  [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]
anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
#stride = [32, 16, 8]

def post_process(res_feature):
    print("len(res_feature)", len(res_feature))
    rect_lst_nms = []
    score_lst = []
    
    for i in range(0, len(res_feature)):
        
        anchor_id = anchors_mask[i]
        anchors_c = [anchors[tmp]  for tmp in anchor_id ]
        
        
        fea_c = res_feature[i]
        print("fea_c is ", fea_c.shape,anchors_c )
        fsize = fea_c.shape[-1]
        stride = 416.0 / fsize
        fea = fea_c.view(1, 3, 5, fsize, fsize )
    
        fea = torch.squeeze(fea)  #      .squeeze_(dim=1)
        print("fea is ", fea.shape)
    
        fea_xy_conf = fea[:, :3, :, :].sigmoid()
        fea_wh  = fea[:, 3:, :, :].detach().exp()
    
        print("fea_wh is ", fea_wh.shape, fea_xy_conf.shape)
    
        #--- from feature compute  pred_box ---

        fea_xy_conf = fea_xy_conf.cpu().detach().numpy()
        fea_wh = fea_wh.cpu().detach().numpy()

        
        # fea_wh nA*wh*fsize*fsize
        for i in range(0, fea_wh.shape[0]):
            for j in range(0, fea_wh.shape[2]):
                for k in range(0, fea_wh.shape[3]):
                    if fea_xy_conf[i][0][j][k] > 0.4:
                        score_lst.append( fea_xy_conf[i][0][j][k] )
                        print("fea_xy_conf[i][1][j][k]", fea_xy_conf[i][1][j][k])
                        bbox_x = fea_xy_conf[i][1][j][k]*(416.0/fsize) + k*(416.0/fsize)
                        bbox_y = fea_xy_conf[i][2][j][k]*(416.0/fsize) + j*(416.0/fsize)
                        bbox_w = ( fea_wh[i][0][j][k] * ( anchors_c[ i ][0]/stride ) ) * stride #*(416.0/fsize) #* 13
                        bbox_h = ( fea_wh[i][1][j][k] * ( anchors_c[ i ][1]/stride ) ) * stride #*(416.0/fsize)  #* 13
                        print("bbox_x, bbox_y, bbox_w, bbox_h ", bbox_x, bbox_y, bbox_w, bbox_h)
                        
                        rect_lst_nms.append([bbox_x - bbox_w/2.0, bbox_y - bbox_h/2,  bbox_x + bbox_w/2.0, bbox_y + bbox_h/2,])

    res = utils.nms(np.array(rect_lst_nms ), 0.4, np.array(score_lst) )

    print("rect_lst is ", res, rect_lst_nms)
    res_nms = []
    for i in range(0,len(rect_lst_nms)):
        if i in res:
             res_nms.append(rect_lst_nms[i])
    return res_nms
                           
    
@torch.no_grad()
def change2eval(m):
    if type(m)== nn.BatchNorm2d :
        m.track_running_stats = False    
    
    
    
conf = {}
model = darknet.YOLOv3(conf,0.7)  
ckpt = torch.load(sys.argv[1], map_location=lambda storage, loc: storage)
model.load_state_dict(ckpt['model'])


model = model.cuda()

cuda = torch.cuda.is_available() and True
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#state = torch.load(sys.argv[1])
print(sys.argv[1])

model.eval()
#if 'model_state_dict' not in state.keys():
#model.load_state_dict( torch.load(sys.argv[1]))

for img_name in os.listdir("../test_data/test_voc_model")[:]:
    if ".jpg"  not in img_name:
        continue
    
    img = cv2.imread('../test_data/test_voc_model/' + img_name, 1)
    print("img_name", img_name)
    
    
    img_size = 416
    
    trans = coco_dataset_p.get_affine_transform((img.shape[1],img.shape[0]), (img_size, img_size))
    img = cv2.warpAffine(img, trans, (img_size,img_size))
    
    #img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)/255.0
    img_raw = img.astype(np.float32)/255.0
    img_raw = img_raw.transpose((2, 0, 1))
    #print(img_raw.shape,img_raw[:,200,:])
    
    #img_raw = np.zeros((1,3,416,416)).astype(np.float32)
    img_th = torch.from_numpy(img_raw).reshape(1,3, img_size, img_size).to(device)
    #print(img_th)
    
    with torch.no_grad():
    
        res_feature = model(img_th)
        #print(res_feature)
        #res_feature = res_feature.sigmoid()

        rect_lst = post_process(res_feature)


        img_show = img.copy()
        for rect in rect_lst:
            bbox_x2 = rect[2]  #rect[0] + rect[2] 
            bbox_y2 = rect[3]  #rect[1] + rect[3] 
            
            #bbox_x2 =  rect[2] 
            #bbox_y2 =  rect[3] 
            cv2.rectangle(img_show,(int(rect[0]), int(rect[1])),( int(bbox_x2), int(bbox_y2) ), (0, 255, 0), 2)
            
        cv2.imwrite( "../test_data/test_yolo_v3_coco_nms_3/debug_" + str(img_name) + ".jpg", img_show )


