import numpy as np
import os
import sys
import torch

import torch.nn as nn

from collections import defaultdict
#from models.yolo_layer import YOLOLayer
sys.path.append("..") 
import utils
import sys
import torch.nn.functional as F
import math
import cv2
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')


def print_execute_time(func):
    from time import time

    # print function 
    def wrapper(*args, **kwargs):
        # begin time
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        # print excute time
        print(f'{func.__name__}() execute time: {end - start}s')
        # return time
        return func_return
    # returns
    return wrapper

# show object confidence feature map and target
#@print_execute_time
def debug_show_feat_map(target, fea_conf, imgs, fea_0, anchors_c, target_true ):
    
    f_s = fea_conf.shape[-1]
    
    target.squeeze_(1)
    #print(" f_s shape ", fea_conf.shape)
    score_pred_show = (fea_conf[0]*255).cpu().detach().numpy().astype(np.uint8)
    target_show = torch.sum(target[0], dim=0)
    
    
    score_gt_show = (target_show[:,:].reshape(f_s, f_s)*255).cpu().detach().numpy().astype(np.uint8)  
    #print(" score_gt_show shape ", score_gt_show.shape)

    score_show = np.zeros((f_s, f_s*4),np.uint8)
    score_show[:,0:f_s]  = score_pred_show[0]
    score_show[:,f_s:f_s*2]  = score_pred_show[1]
    score_show[:,f_s*2:f_s*3]  = score_pred_show[2]
    score_show[:,f_s*3:f_s*4]  = score_gt_show#score_pred_show[3]
    if f_s <=26:
        score_show = cv2.resize(score_show,(f_s*4*10,f_s*10))
    else:
        score_show = cv2.resize(score_show,(f_s*4*5,f_s*5))
    
    target_0 = target[0]
    #print(" target_show shape ", target_0.shape)
    img_show_1 = (imgs[0].reshape(3,416,416)*255.0).cpu().detach().numpy().astype(np.uint8).transpose((1, 2, 0))
    img_show = img_show_1.copy().reshape(416,416,3).astype(np.uint8) #np.zeros((416, 416, 3),dtype=np.uint8)
    
    #print("fea_0 shape is ", fea_0.shape)
    fea_xy = fea_0[ :, 1:3, :, :].sigmoid()
    fea_xy = fea_xy.cpu().detach().numpy()
    fea_wh = fea_0[ :, 3:, :, :].cpu().detach().exp().numpy()
    
    #print("target_0 ",torch.sum(target_0 , dim=0))
    for i in range(0, fea_0.shape[0]): #anchor
        for j in range(0, fea_0.shape[2]):  #h
            for k in range(0, fea_0.shape[3]): #w
                #print(target_0[i][j][k])
                if target_0[i][j][k] > 0:
                    #if f_s == 26:
                        
                    #print("fea_xy[i][1][j][k] ", fea_xy[i][1][j][k], fea_xy[i][0][j][k],i,j,k, 416.0/f_s )
                    bbox_x = fea_xy[i][0][j][k]*(416.0/f_s) + k*(416.0/f_s)
                    bbox_y = fea_xy[i][1][j][k]*(416.0/f_s) + j*(416.0/f_s)
                    
                    #print("fea_wh[i][1][j][k]  ", fea_wh[i][1][j][k] , fea_wh[i][0][j][k]  )
                    bbox_w =  fea_wh[i][0][j][k] * anchors_c[ i ][0] * (416.0/f_s) #*(416.0/fsize) #* 13
                    bbox_h =  fea_wh[i][1][j][k] * anchors_c[ i ][1] * (416.0/f_s) #*(416.0/fsize) #* 13
                    #print("x,y,w,h ", int(bbox_x - bbox_w//2), int(bbox_y - bbox_h//2), int(bbox_x + bbox_w//2), int(bbox_y + bbox_h//2) )
                    y2,x2 = min(int(bbox_y + bbox_h//2),416),min( int(bbox_x + bbox_w//2), 416)
                    cv2.rectangle(img_show, (  max(0, int(bbox_x - bbox_w//2)), max(0,int(bbox_y - bbox_h//2)) ), (x2 ,y2 ), (0,0,255), 2 )
    
    for item in target_true[0]:
        #print(item)
        if torch.sum(item, dim=0).item()>0:
            xc = item[0]*416
            yc = item[1]*416
            wid = item[2]*416
            heig = item[3]*416
            #print(" xc,yc,wid,heig", xc,yc,wid,heig)
            cv2.rectangle(img_show, (  max(0, int(xc - wid//2)), max(0,int(yc - heig//2)) ), ( int(xc + wid//2) ,int(yc + heig//2) ), (0,255,0), 2 )
    
    return score_show,img_show
    

class YOLOv3_layer(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, S, B, l_coord, l_noobj, layer_num, nc=1):
        super(YOLOv3_layer, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        #self.anchors = [ [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],[7.88282, 3.52778], [9.77052, 9.16828] ]#base anchor
        self.anchors = [[10,13],  [16,30],  [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]
        self.anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
        self.stride = [32, 16, 8]
        self.in_ch_lst = [1024, 512, 256]
        self.layer_num = layer_num
        self.in_ch = self.in_ch_lst[layer_num]
        self.nC = nc
        self.nA = len(self.anchors_mask[layer_num])
        self.bce_loss = nn.BCELoss(size_average=False)
        
        self.conv = nn.Conv2d(in_channels = self.in_ch, out_channels = self.nA * (self.nC + 5), kernel_size = 1, stride = 1, padding = 0)
        
    #target is [n,xc,yc,w,h] after normlization
    #input is fea [N, C, H, W]
    #@print_execute_time
    def forward(self, fea, target ,imgs):
        #print("target is ", target)
        
        fea = self.conv(fea)
        
        if target is None:  # not training
            #fea[..., :4] *= self.stride
            return fea
        train = target is not None
        
        layer_num = self.layer_num
        num_class = 0
        bs = fea.shape[0]
        fsize =  fea.shape[2]
        n_ch =  self.nC  + 5
        n_A = 3
        stride = self.stride[layer_num]
        
        # assert
        assert(fea.shape[1] == n_A*n_ch)
        fea = fea.view(bs, n_A, n_ch, fsize, fsize)
        
        conf_neg_mask, conf_pos_mask, target_conf, target_loc, loc_coefficient, target_cls = self.get_loss_target(target, fea, layer_num)
        
        
        fea_conf = fea[:, :, 0, :, :].sigmoid()
        loc_coefficient = loc_coefficient.expand_as(fea[:, :, 3:5, :, :])
 
        
        fea_xy = fea[:, :, 1:3, :, :].sigmoid()#*loc_coefficient
        fea_wh = fea[:, :, 3:5, :, :] #* loc_coefficient
        
        fea_cls = fea[:, :, 5:, :, :].sigmoid()        
        
        target_loc_xy = target_loc[:,:,:2,:,:]#*loc_coefficient
        target_loc_wh = target_loc[:,:,2:,:,:]#*loc_coefficient


        #---------------------------begin debug show ------------------------------------------------------
        debug_show = True
        fea_0 = fea[0]
        
        anchors_c_ind =  self.anchors_mask[layer_num]
        anchors_c = self.anchors
        
        anchors_c_debug  = [[anchors_c[i][0]*1.00/self.stride[layer_num], anchors_c[i][1]*1.00/self.stride[layer_num]] for i in  anchors_c_ind]
        
        if debug_show and layer_num == 0 :
            score_show_0, img_show_0 = debug_show_feat_map(target_conf, fea_conf, imgs, fea_0, anchors_c_debug, target)
            cv2.imwrite("score_show_yolo_v3_l0.png", score_show_0)
            cv2.imwrite("img_show_00.png", img_show_0)
            
        if debug_show and layer_num == 1 :
            score_show_1, img_show_1 = debug_show_feat_map(target_conf, fea_conf, imgs, fea_0, anchors_c_debug, target)
            cv2.imwrite("score_show_yolo_v3_l1.png", score_show_1)
            cv2.imwrite("img_show_11.png", img_show_1)
        if debug_show and layer_num == 2 :
            score_show_2, img_show_2 = debug_show_feat_map(target_conf, fea_conf, imgs, fea_0, anchors_c_debug, target)
            cv2.imwrite("score_show_yolo_v3_l2.png", score_show_2)
            cv2.imwrite("img_show_22.png", img_show_2)
        #input("1")
        
        #--------------------------- end debug show ------------------------------------------------------

        
        #print("conf_neg_mask shape is ", (conf_pos_mask>0).sum())
        #print("target_conf shape is ", (target_conf>0).sum())
        
        
        conf_neg_mask_b = conf_neg_mask > 0
        conf_pos_mask_b = conf_pos_mask > 0
        
        #print("fea_conf shape is ", fea_conf.shape)
        fea_conf_use = fea_conf[conf_pos_mask_b]
        tar_conf_use = target_conf[conf_pos_mask_b]
        #print("fea_conf_use  is ", target_conf[conf_neg_mask_b], fea_conf[conf_neg_mask_b]) #conf_neg_mask_b
        
        obj_conf_loss = F.mse_loss( tar_conf_use, fea_conf_use, size_average = False )
        #print("tar_conf_use is ",  tar_conf_use.shape,  fea_conf_use.shape)
        noobj_conf_loss = F.mse_loss( target_conf[conf_neg_mask_b], fea_conf[conf_neg_mask_b] , size_average=False)
        
        obj_wh_loss = F.mse_loss( fea_wh, target_loc_wh,  size_average=False)
        obj_xy_loss = F.mse_loss( fea_xy, target_loc_xy,  size_average=False )
        # add object cls
        #print(fea_cls.shape, target_cls.shape)
        obj_cls_loss = self.bce_loss( fea_cls, target_cls)
        
        #print("obj_wh_loss is ,", (obj_conf_loss*5).item() , noobj_conf_loss.item() ,obj_wh_loss.item() ,obj_xy_loss.item() )
       
        conf_loss = obj_conf_loss*5 + noobj_conf_loss + obj_wh_loss + obj_xy_loss + obj_cls_loss*0
       
        return conf_loss, obj_conf_loss*5, noobj_conf_loss,  obj_wh_loss + obj_xy_loss #, obj_cls_loss*0

    
    def get_loss_target(self, target, pred, layer_num ):
        
        f_size = int(416/self.stride[layer_num])
        if layer_num >= 3:
            layer_num = 2
            
        anchors_c_ind =  self.anchors_mask[layer_num]
        #print("anchors_c_ind  is ", anchors_c_ind)
        anchors_all = [[self.anchors[i][0]*1.00/self.stride[layer_num], self.anchors[i][1]*1.00/self.stride[layer_num]] for i in  range(0, len(self.anchors) )]
        
        # anchor size is relative to current feature map
        anchors_c = [ anchors_all[i] for i in  anchors_c_ind]
        
        #print("anchors_c  is ", anchors_c, anchors_all , pred.shape, layer_num, f_size)
        nB = pred.shape[0]
        nA = len(anchors_c_ind)
        
        #print("target  is ", target.shape)
        device = pred.device
        
        conf_pos_mask = torch.zeros(nB, nA, f_size, f_size, requires_grad=False, device=device)
        conf_neg_mask = torch.ones(nB, nA, f_size, f_size, requires_grad=False, device=device)
        
        target_conf = torch.zeros(nB, nA, f_size, f_size, requires_grad=False, device=device)
        loc_mask = torch.zeros(nB, nA, f_size, f_size, requires_grad=False, device=device)
        
        loc_coefficient = torch.zeros(nB, nA, 1, f_size, f_size, requires_grad=False, device=device)
        
        target_location = torch.ones(nB, nA, 4, f_size, f_size, requires_grad=False, device=device)
        target_cls = torch.ones(nB, nA, self.nC, f_size, f_size, requires_grad=False, device=device)
        
        for b in range(nB):
            #compute pred box
            #pred_boxes = torch.zeros(5*f_size*f_size, 4, dtype=torch.float, device=device)
            
            #---predict bbox and gt iou compute---
            #pred rect location is about feature map size
            #pred_boxes[:,0] = pred[b, :, 1, :, :]
            #pred_boxes[:,1] = pred[b, :, 2, :, :]
            #pred_boxes[:,2] = pred[b, :, 3, :, :]
            #pred_boxes[:,3] = pred[b, :, 4, :, :]
            
            #--- end predict bbox and gt iou compute---
            
            #gt_org  value is. (0-1)
            #**************must use .clone(),because, equals is not deep copy,target will be change**************
            gt_org = target[b].clone()
            mask_gt_org = gt_org.gt(0)  # tensor.gt is greater than 0
            
            gt_num = (mask_gt_org == True).sum(dim=0)
            gt_num_n = torch.max(gt_num)
            gt_org = gt_org[:gt_num_n,:]
            #print("gt_num_n is ", gt_num_n )

            
            #gt_wh = torch.zeros([gt_num_n, 4], requires_grad=False, device=device)
            gt_wh = gt_org[:,0:4].clone()
            gt_wh[:,:2] = 0
            
            anchors = torch.zeros([len(anchors_all), 4], requires_grad=False, device=device)
            anchors[:,2:] = torch.from_numpy(np.array(anchors_all))  #anchors x y w h is about layer size(13,26,52)
            
            gt_wh = gt_wh.float()
            gt_wh = gt_wh * 416.0/self.stride[layer_num]  #convert width and height size about layer size(13,26,52)
            #print("gt_org  before is ", gt_org)
            gt_org[:,0:4] = gt_org[:, 0:4] * 416.0/self.stride[layer_num] 
            
            iou_gt_anchors = utils.bbox_ious(gt_wh, anchors)
            anchor_id = torch.max(iou_gt_anchors, dim=1)

            for j in range(0, len(gt_org)):
                #anchor_id[1] is true max id (xiabiao)
                id_tmp = anchor_id[1][j]
                #print("id_tmp",id_tmp , anchors_c_ind, f_size)
                if id_tmp  not in anchors_c_ind:
                    continue
                else:
                    #compute anchor_id in  this output layer[ number is  0 1 2 ] 
                    anchor_id_t = id_tmp - anchors_c_ind[0]
                    #print("anchor_id_t is ", anchors_c_ind[0],id_tmp)
                    
                # gt_org  coordinate should about  feature map (13,26,52)
                #ind_i and ind_j is about  feature map (13,26,52)
                ind_i = min(max(int(gt_org[j][1]), 0), f_size - 1) # gt_org[j][0] is ith row
                ind_j = min(max(int(gt_org[j][0]), 0), f_size - 1) # gt_org[j][1] is ith col

                conf_neg_mask[b][anchor_id_t][ind_i][ind_j] = 0#[ind_i][ind_j]
                conf_pos_mask[b][anchor_id_t][ind_i][ind_j] = 1
                target_conf[b][anchor_id_t][ind_i][ind_j] = 1
                
                #location loss coefficient, (2-bbox_w*bbox_h) 
                #13 need change
                loc_coefficient[b][anchor_id_t][0][ind_i][ind_j] = 2.0 - (gt_org[j][3]*1.0/f_size) * (gt_org[j][2]*1.0/f_size)
                
                #location( about feature map (13,26,52)) gt(x and y) , gt is offset coordinates relative x y
                target_location[b][anchor_id_t][1][ind_i][ind_j] = gt_org[j][1] - ind_i
                target_location[b][anchor_id_t][0][ind_i][ind_j] = gt_org[j][0] - ind_j
                
                
                #location gt(w and h is about feature map (13,26,52)) , w h is about anchor_w*e_tw, tw is width gt
                elem_ww  = max(gt_org[j][2] / anchors_c[anchor_id_t][0], 0.000001)
                elem_hh  = max(gt_org[j][3] / anchors_c[anchor_id_t][1], 0.000001)
                
                
                target_location[b][anchor_id_t][2][ind_i][ind_j] = math.log( elem_ww )
                target_location[b][anchor_id_t][3][ind_i][ind_j] = math.log( elem_hh ) #hh / self.anchors[anchor_id[1][j]][0] )#gt_org[j][2]
                
                # cls target  loss 
                target_cls[b][anchor_id_t][ int(gt_org[j][4]-1) ][ind_i][ind_j]  = 1
                
                #input('1')
            

        return conf_neg_mask, conf_pos_mask, target_conf, target_location, loc_coefficient, target_cls
            

    #@print_execute_time
    def target2bbox_lst(self, fea):
        if len(fea.shape) == 4:
            #print("test.......")
            fea.squeeze_(0)
            #print("test.......",fea.shape)
        rect_lst = []
        ratio_fea = 416.0 / self.stride[self.layer_num]
        #print("target.shape is ", fea.shape)
        for i in range(fea.shape[0]):
            for j in range(fea.shape[1]):
                if fea[i][j][0] > 0.8:
                    
                    #print(fea[0][i][j][0])
                    bbox_x = fea[i][j][1] * ratio_fea #j*(416.0/13) + fea[i][j][1]*(416.0/13) #fea[0][i][j][1]*416 #j*(416.0/13) + fea[0][i][j][1]*(416.0/13)
                    bbox_y = fea[i][j][2] * ratio_fea #i*(416.0/13) + fea[i][j][2]*(416.0/13) #fea[0][i][j][2]*416 #i*(416.0/13) + fea[0][i][j][2]*(416.0/13)
                    bbox_w =  fea[i][j][3]* ratio_fea
                    bbox_h =  fea[i][j][4]* ratio_fea
                    
                    rect_lst.append([bbox_x, bbox_y, bbox_w, bbox_h])
        return rect_lst

            