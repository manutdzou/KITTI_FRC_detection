# 本项目以VGG_CNN_M_1024为例，训练数据为KITTI

## 本项目所有路径为/home/bsl/KITTI_FRC_detection,在不同电脑上需要修改路径

### 1. 网络模型

网络模型在/models/VGG_CNN_M_1024/faster_rcnn_alt_opt中，4个training stage,运行draw_net.sh可以画出每个步骤的网络图，需要在draw_net.py中添加正确的pycaffe路径

### 2. 数据

下载KITTI数据，解压到data中，数据目录为data/training/image_2/...
ImageList_Version_S_AddData.txt为所有数据list,ImageList_Version_S_GT_AddData.txt为所有数据的不同类别的groundtruth,格式为 image_path object_num class1_num coordinates class2_num coordinates... 如果某一类数量为0则class_num为0，coordinates空缺，例如training/image_2/000240.png 1 1 567.32 177.55 609.97 215.63 0 0 表示000240.png一共有一个物体，第一类物体一个x_left,y_left,x_right,y_right为567.32 177.55 609.97 215.63，第二类和第三类没有物体。所有数据标签做好以后生成train和val数据集，利用Matlab的split_data.m生成KITTI_train_list.txt,KITTI_gt_train.txt和KITTI_val_list.txt,KITTI_gt_val.txt，选用70%作为训练数据，30%作为测试数据。

### 3. 模型

在imagenet_models下载fine_tune的模型VGG_CNN_M_1024.v2.caffemodel,下载命令/scripts/fetch_imagenet_models.sh. 在lib/rpn/generate_anchors.py中设置7个ratios和10个scales共70个anchors，所以模型文件中rpn_cls_score的num_output为140  2(bg/fg) * 70(anchors),rpn_bbox_pred的num_output为280  4 * 70(anchors).需要注意rpn_cls_prob_reshape层的channels应该设为和rpn_cls_score的num_output一致。

### 4. 生成数据库

数据库接口在lib/datasets/中，，注意修改绝对路径。不同于原有的VOC数据格式接口读取xml，该数据接口实现将txt的图片路径和图片真值框列表读入.pkl

### 5. 算法评估工具

算法评估工具在VOCdevkit-matlab-wrapper中，写了一个KITTI的评估工具，衡量标准为AP值，该工具集成到了python程序中由Python调用matlab.也可以单独由检测生成的/data/results/中的./txt利用main.m直接产生结果。在data/results中可以生成对应类别的所有检测结果图片，TP和FP的结果图片以供分析。

detection_eval.m为算法评估主程序，这里定义了三个类classes={'car','person','bike'}，minoverlap=0.2，建议在一般情况下将读写图片注释掉，因为读写过程比较慢。

### 6. tools里自己写的工具说明

demo_location.py主要将检测结果记录在txt里，这个接口是为我们的私有数据写的，在这里没有使用。

demo_show.py主要实现将检测结果和真值框同时显示在图片上并将图片保存到data/results/show里面。由于show文件夹没有写成自动建立，需要自建（实在没时间做程序优化啊，见谅啊）。

demo_for_video.py主要实现将视频帧存储的连续图片检测然后做成视频格式，视频格式可自选，默认格式大小最适合。

demo_video_for_video.py顾名思义是读视频然后检测写入视频咯，注意需要将视频放在申明的位置。

train_debug.py和test_net_debug.py主要为了在pycharm中调试使用，此时需要将参数全部写入。

### 7. 参数设置

参数设置主要在lib/fast_rcnn/config.py中__C.TRAIN.SCALES = (370,300,250,200,150)表示训练多尺度的样本，__C.TRAIN.MAX_SIZE = 1250表示样本最大边长，其他batch_size,IOU等都可以在这里设置。

### 8. 训练

可以直接在终端运行train.sh调用experiments/scripts/faster_rcnn_alt_opt.sh训练网络，经过4个80000次训练获得网络模型和测试结果。你将最终获得APs: Car:88.6, Person:73.0, Bike: 69.5

If you have any problems or find some bugs, please contact with manutdzou@126.com and feel free. Thank you again for your observation.
