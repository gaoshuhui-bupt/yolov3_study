import os
import pycocotools.coco as coco
from   pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask
import numpy as np
import cv2
import json

def draw_masks(img,masks):
    colors_mask = np.array([[0,255,0],[255,0,0],[0,0,255],[255,255,0],[0,255,255],[125,0,0],[0,125,0],[125,0,125],[100,50,0],[50,100,0],[0,50,100]], dtype=np.uint8)
    w,h = img.shape[1],img.shape[0]
    for i in range(len(masks)):
        for r in range(h):
            for c in range(w):
                if masks[i][r,c] > 0 :
                    img[r,c] = img[r,c]*0.4 +  colors_mask[i%11]*0.6
    return img

def draw_segs(img,segs_list):
    colors_mask = np.array([[0,255,0],[255,0,0],[0,0,255],[255,255,0],[0,255,255],[125,0,0],[0,125,0],[125,0,125],[100,50,0],[50,100,0],[0,50,100]], dtype=np.uint8)
    contours_list = []
    for segs in segs_list : # each segs a object
        for seg in segs :
            contours = np.zeros((len(seg)//2,1,2),np.int)
            for i in range(0,len(seg)//2):
              contours[i,0,0]   = int(seg[2*i + 0])
              contours[i,0,1]   = int(seg[2*i + 1])
            contours_list.append(contours)
    cv2.drawContours(img,contours_list,-1,(0,0,255),1)   
    return img

coco_base_path = "../../../data/coco_org/"
coco_ins = coco.COCO('./data/annotations/instances_train2017.json')
#instances_val2017.json
#coco_ins = coco.COCO('./data/annotations/instances_val2017.json')
img_dir = coco_base_path + "train2017/"

save_txt_path = "/data01/gaosh/my_task/data/coco_one_label/"
images = coco_ins.getImgIds()
print(len(images))
for index in range(0,len(images)):
    if index % 1000 == 0:
        print(index)
    img_id = images[index]
    img_name = coco_ins.loadImgs(ids = [img_id])[0]['file_name']
    ann_ids = coco_ins.getAnnIds(imgIds=[img_id])
    anns = coco_ins.loadAnns(ids=ann_ids)
    #print(anns)
    
    img = cv2.imread(os.path.join(img_dir,img_name), 1)
    height, width,_ = img.shape
    height,width = img.shape[0],img.shape[1]
    #print(os.path.join(img_dir,img_name))
    num_objs = len(anns)
    segs_list = []
    
    have_person = False
    #if img_name != "000000124442.jpg":
    #    continue
    
        
    for k in range(num_objs):
        ann  = anns[k]
        #if int(ann['category_id']) == 1 and ann['iscrowd'] != 1:
        if  ann['iscrowd'] != 1 and int(ann['category_id']) == 1 :
            have_person = True
            txt_path = open(save_txt_path + img_name + ".txt",'w' )
            break
        
    
    for k in range(num_objs):
        ann  = anns[k]
        box = ann["bbox"] 
        cls_id  = ann['category_id']
        if ann['iscrowd'] == 1:
            continue
        if cls_id != 1 :
            continue
        segs = ann["segmentation"]

        x_list = []
        y_list = []
        for seg in segs :
            for i in range(0,len(seg)//2):
                x_list.append(seg[2*i + 0])
                y_list.append(seg[2*i + 1])
        box[0] = min(x_list)
        box[1] = min(y_list)
        box[2] = max(x_list)+1
        box[3] = max(y_list)+1
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int( box[2]), int( box[3])),(0,255,0),2)
        cx = ((box[0] + box[2])/2.0)/width*1.00
        cy = ((box[1] + box[3])/2.0)/height*1.00
        w = (box[2] - box[0])/width*1.00
        h =  (box[3] - box[1])/height*1.00
        txt_path.write(str(cx) + " " + str(cy) + " " + str(w) + " " + str(h)  + " "+ str(cls_id) + "\n" )
    txt_path.close()   
        #cv2.imwrite(os.path.join("./anno_debug",img_name),img)
    
    """
    

    
    # retrieve every img, every img generate two imgs
    
    for k in range(num_objs):
        ann  = anns[k]
        if int(ann['category_id']) == 1:
            have_person = True
            #txt_path = open(save_txt_path + img_name + ".txt",'w' )
            break
    
    for k in range(num_objs):
        ann  = anns[k]
        print(ann)
        box = ann["bbox"] 
        
        segs = ann["segmentation"]
        print(ann['category_id'])
        if int(ann['category_id']) == 1 :
            #txt_path.write( str(ann['category_id']) + " ")
            cx = round((box[0] + box[2]/2.0)/width*1.00,6)
            cy = round((box[1] + box[3]/2.0)/height*1.00, 6)
            w = round( box[2]/width*1.00, 6)
            h = round( box[3]/height*1.00, 6)
            print(cx,cy,w,h)
            
            #txt_path.write(str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + "\n" )
        
        
            cv2.rectangle(img, (int(box[0]),int(box[1 ]) ),( int(box[0]) + int(box[2]),int(box[1])+int(box[3]) ),(0,255,0) ,2  )
            cv2.imwrite(os.path.join("./anno_debug",img_name),img)
        #segs_list.append(segs)
    #img = draw_segs(img,segs_list)
    #cv2.imwrite(os.path.join("./anno_debug",img_name),img)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    """
