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
from input import coco_dataset_v3
import darknet
import time

from torch.utils.tensorboard import SummaryWriter

# 000. bs device tensor type ...
batch_size = 16
torch.cuda.set_device(2)
cuda = torch.cuda.is_available() and True
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 001. dataset train 
dataset = coco_dataset_v3.COCODataset_Person(model_type='YOLO',
                  data_dir='../../../../../../../my_task/data/coco_org/train2017/',#coco_person/coco_2017_person_yolo_format/train/images/',
                  data_dir_labels='../../../../../../../my_task/data/coco_one_label/', 
                  img_size = 416,
                  augmentation = False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 001. dataset test 
"""
dataset_test = coco_dataset_org.COCODataset_Person(model_type='YOLO',
                  data_dir='../../../../../my_task/data/coco_person/coco_2017_person_yolo_format/train/images/',
                  img_size=416,
                  augmentation='')

dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
"""

# 002. model and parameters
conf = {}
model = darknet.YOLOv3(conf, 0.7 )
model = model.cuda()
model.train()
# visualize loss and lr use tensorboard 
writer = SummaryWriter('./log')

# 003. lr train and test optim
base_lr = 0.001
#lambda1 = lambda epoch: (epoch / 4000) if epoch < 4000 else 0.5 * (math.cos((epoch - 4000)/(100 * 1000 - 4000) * math.pi) + 1)

epochs = 100
lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine from yolo v5

optimizer = torch.optim.Adam(model.parameters(), lr = base_lr )
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) 

# 004 .load checkpoint 
resume = True
start_epoch = 0
if resume:
    checkpoints_name = 'checkpoints/yolo_v3_step2200_new_aug_cls.pth'
    #base_lr = 0.001
    if os.path.isfile(checkpoints_name):
        checkpoint = torch.load(checkpoints_name)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer.state_dict()['param_groups'][0]['lr']
        print("===> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        
# 005. train

        
for epoch in range(start_epoch, epochs):
    loss_all = 0
    st = time.time()
    for i, data in enumerate(dataloader, 0):
        if cuda:
            imgs = data[0].cuda()
            targets = data[1].cuda()
        else:
            imgs, targets = data[0], data[1]
            
        loss,  loss_element  = model(imgs, targets, imgs )
        #t =  model(imgs, targets, imgs )
        #print("ttttttttt", t)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        #scheduler.step()
        
        # output epoch loss and step loss 
        #print("loss_element", len(loss_element),loss_element[0][0].item())
        #print("loss_element shape is ", len(loss_element), loss_element[0])
        
        obj_conf_loss = sum(loss_element[:][0])
        obj_loc_loss =  sum(loss_element[:][2])
        noobj_conf_loss =  sum(loss_element[:][1])
        cls_loss =  sum(loss_element[:][-1])
        
        #print("loss   is. ",  loss_element[:][-1])
        #print("loss  ,,, is. ",    loss_element[:][2])
        
        loss_all += loss.item()
        loss_tr = loss.item()
        new_iter = len(dataloader) * epoch + i
        
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=new_iter, walltime=None)
        writer.add_scalar('loss_all', loss_all*1.0/(i+1), global_step=new_iter, walltime=None)
        writer.add_scalar('obj_conf_loss', obj_conf_loss.item(), global_step=new_iter, walltime=None)
        writer.add_scalar('obj_loc_loss', obj_loc_loss.item(),  global_step=new_iter, walltime=None)
        writer.add_scalar('noobj_conf_loss', noobj_conf_loss.item(), global_step=new_iter, walltime=None)
        #writer.add_scalar('obj_cls_loss', cls_loss.item(), global_step=new_iter, walltime=None)
        
        
        if i%100==0 and i > 0:
            #print('loss', loss_all/(i+1)*1.00)
            end = time.time()
            print('loss', loss_tr*1.00 , loss_all/(i+1)*1.00, i+1, end-st)
            model_save_path = "checkpoints/yolo_v3_step" + str(i).zfill(4) + "_new_aug_cls.pth"
            state_model = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state_model, model_save_path)
            
    print('epoch:{}    train_loss:{}  lr:{}'.format(epoch+1, loss_all/len(dataloader), optimizer.state_dict()['param_groups'][0]['lr']))
    
    if epoch%2 == 0 and epoch > 0:
        #model_save_path = "checkpoints/yolo_v1_" + str(epoch).zfill(4) + ".pth"
        #torch.save(model, model_save_path)
        
        model_save_path = "checkpoints/yolo_v3_" + str(epoch).zfill(4) +  "_new_aug_cls.pth"
        state_model = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state_model, model_save_path)
    scheduler.step()
        
    
    #if epoch%5==0:
