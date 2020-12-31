import numpy as np
import os
import sys

import torch
#import resnet_v2
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.autograd import Variable
import yolo_v3_loss
import torch.optim as optim
import cv2
import torch.nn.functional as F
import sys
sys.path.append("..") 
from input import coco_dataset_org
import darknet
import utils

from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(2)
device = torch.device("cuda:2")

anchors = [[10,13],  [16,30],  [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]
anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]


   
def target2bbox_lst( fea):
    #print("fea shape is :", fea.shape)
    if True:
        #print("test.......")
        #print("test.......",fea.shape)
        fea = np.squeeze(fea) #fea.squeeze_(0)
        #print("test.......",fea.shape)
        
    rect_lst = []
    ratio_fea = 416.0 / fea.shape[-1]
    ratio_fea = 416.0
    print("target.shape is ", ratio_fea,fea.shape)
    for i in range(fea.shape[0]):
        for j in range(fea.shape[1]):
            if fea[i][j][0] > 0.8:
                    
                #print(fea[0][i][j][0])
                bbox_x = fea[i][j][1] * 416 #j*(416.0/13) + fea[i][j][1]*(416.0/13) #fea[0][i][j][1]*416 #j*(416.0/13) + fea[0][i][j][1]*(416.0/13)
                bbox_y = fea[i][j][2] * 416 #i*(416.0/13) + fea[i][j][2]*(416.0/13) #fea[0][i][j][2]*416 #i*(416.0/13) + fea[0][i][j][2]*(416.0/13)
                bbox_w =  fea[i][j][3]* 416
                bbox_h =  fea[i][j][4]* 416

                rect_lst.append([bbox_x - bbox_w/2.0, bbox_y - bbox_h/2.0, bbox_x + bbox_w/2.0, bbox_y + bbox_h/2.0])
    return rect_lst

def post_process(res_feature):
    #print("len(res_feature)", len(res_feature))
    rect_lst_nms = []
    score_lst = []
    
    for i in range(0, len(res_feature)):
        
        anchor_id = anchors_mask[i]
        anchors_c = [anchors[tmp]  for tmp in anchor_id]
        
        
        fea_c = res_feature[i]
        #print("fea_c is ", fea_c.shape,anchors_c )
        fsize = fea_c.shape[-1]
        stride = 416.0 / fsize
        fea = fea_c.view(1, 3, 5, fsize, fsize )
    
        fea = torch.squeeze(fea)  #      .squeeze_(dim=1)
        #print("fea is ", fea.shape)
    
        fea_xy_conf = fea[:, :3, :, :].sigmoid()
        fea_wh  = fea[:, 3:, :, :].detach().exp()
    
        #print("fea_wh is ", fea_wh.shape, fea_xy_conf.shape)
    
        #--- from feature compute  pred_box ---

        fea_xy_conf = fea_xy_conf.cpu().detach().numpy()
        fea_wh = fea_wh.cpu().detach().numpy()

        
        # fea_wh nA*wh*fsize*fsize
        for i in range(0, fea_wh.shape[0]):
            for j in range(0, fea_wh.shape[2]):
                for k in range(0, fea_wh.shape[3]):
                    if fea_xy_conf[i][0][j][k] > 0.4:
                        score_lst.append( fea_xy_conf[i][0][j][k] )
                        #print("fea_xy_conf[i][1][j][k]", fea_xy_conf[i][1][j][k])
                        bbox_x = fea_xy_conf[i][1][j][k]*(416.0/fsize) + k*(416.0/fsize)
                        bbox_y = fea_xy_conf[i][2][j][k]*(416.0/fsize) + j*(416.0/fsize)
                        bbox_w = ( fea_wh[i][0][j][k] * ( anchors_c[ i ][0]/stride ) ) * stride #*(416.0/fsize) #* 13
                        bbox_h = ( fea_wh[i][1][j][k] * ( anchors_c[ i ][1]/stride ) ) * stride #*(416.0/fsize)  #* 13
                        #print("bbox_x, bbox_y, bbox_w, bbox_h ", bbox_x, bbox_y, bbox_w, bbox_h)
                        
                        rect_lst_nms.append([bbox_x - bbox_w/2.0, bbox_y - bbox_h/2,  bbox_x + bbox_w/2.0, bbox_y + bbox_h/2])

    res = utils.nms(np.array(rect_lst_nms ), 0.4, np.array(score_lst) )

    #print("rect_lst is ", res, rect_lst_nms)
    res_nms = []
    score_nms = []
    for i in range(0,len(rect_lst_nms)):
        if i in res:
            res_nms.append(rect_lst_nms[i])
            score_nms.append(score_lst[i])
            
    return res_nms,score_nms

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

#../../../../../my_task/data/coco_person_label/test/

dataset_test = coco_dataset_org.COCODataset_Person(model_type='YOLO',
                  data_dir='../../../../../my_task/data/coco_person/coco_2017_person_yolo_format/val/images/',
                  data_dir_labels = '../../../../../my_task/data/coco_person_label/test/',
                  img_size=416,
                  augmentation=False)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

all_gt = 0
all_tp = 0
all_fp = 0

#while(1):
if True:
    for i, data in enumerate(dataloader_test, 0):
            #input('1')
            if cuda:
                imgs = data[0].cuda()
                targets = data[1].cuda()
            else:
                imgs, targets = data[0], data[1]

            #with torch.no_grad():
            res_feature = model(imgs)
            

            rect_lst,score_lst = post_process(res_feature)
            #print(data[0])
            img_show = torch.squeeze(data[0])
            
            img_show = img_show * 255
            img_show = img_show.cpu().detach().numpy().copy()
            
            img_show = img_show.transpose((1, 2, 0))
            img_show = img_show.astype(np.uint8) 
            img = img_show
            
            
            # targets lst
            targets = targets.cpu().detach().numpy().copy()
            targets_bbox_lst = target2bbox_lst(targets)
            print("len(rect_lst) ", rect_lst,targets_bbox_lst )
            
            if len(rect_lst) != 0:
                
                lst_tp_fp = utils.voc_eval(np.array(rect_lst), np.array(score_lst), np.array(targets_bbox_lst))
                print("lst_tp_fp is ", lst_tp_fp )
                
                all_tp +=   lst_tp_fp[0]
                all_fp +=   lst_tp_fp[1]
                all_gt +=   lst_tp_fp[2]
            else:
                all_gt +=   len(targets_bbox_lst)
                
            
            
            for gt_i in targets_bbox_lst:
                img = cv2.UMat(img).get()
                cv2.rectangle(img , ( int(gt_i[0]), int(gt_i[1]) ),( int(gt_i[2]), int(gt_i[3]) ), (0, 0, 255), 2 )
            
            for rect in rect_lst:
                bbox_x2 = rect[2]  #rect[0] + rect[2] 
                bbox_y2 = rect[3]  #rect[1] + rect[3] 
                
                img = cv2.UMat(img).get()
                
                cv2.rectangle(img , ( int(rect[0]), int(rect[1]) ),( int(bbox_x2), int(bbox_y2) ), (0, 255, 0), 2 )


            #cv2.imwrite( "../test_data/test_yolo_v3_coco_nms_aug/debug_" + str(i).zfill(5) + ".jpg", img)

print("all_tp, all_fp, all_gt", all_tp, all_fp, all_gt)
print("recall, precision ",all_tp*1.0/all_gt, all_tp*1.0/(all_tp + all_fp) )

