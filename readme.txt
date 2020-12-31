使用说明：
（1）process_data是处理coco数据的脚本，将crowed去除掉了，会在指定文件夹下生成图片名字为文件名称的txt文件，格式为xc，yc，width，heigh，class_id,分隔符号为\t
(2)train_yolo_v3位训练脚本，需要将图片的路径和标签的路径给到COCODataset_Person，示例如下：
dataset = coco_dataset_v3.COCODataset_Person(model_type='YOLO',
                  data_dir='../../../../../../../my_task/data/coco_org/train2017/',#coco_person/coco_2017_person_yolo_format/train/images/',
                  data_dir_labels='../../../../../../../my_task/data/coco_one_label/', 
                  img_size = 416,
                  augmentation = False)
（3）log下保存了tensorboard需要的文件，每2轮在checkpoints下保存权重
（4）demo_yolo_v3尾单张图片预测，test_yolo_v3为批量图片预测，并计算precision和recall的脚本