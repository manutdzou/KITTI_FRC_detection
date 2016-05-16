#!/usr/bin/env python
# --------------------------------------------------------
# kitti tool
# Copyright (c) 2016 Chao Chen
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao Chen
# change the KITTI label to our txt labels
# --------------------------------------------------------

import os

def getLabelFilename(img_filename):
     vector_string = img_filename.split('.'); 
     return vector_string[0] + '.txt'





def parse_line(img_line, label_file):
    count_p = 0
    count_c = 0  
    count_cyc = 0
    p_flag = 0
    c_flag = 0
    cyc_flag = 0
    bbox_car = []
    bbox_per = []
    bbox_cyc = []
    file = open('../data_object_image_2/' + label_file, 'r')
    for line in file.xreadlines():
        line = line.strip('\n')
        vector_str = line.split(' ')
        #print vector_str
        if ("Car" == vector_str[0] ):
            count_c += 1
            for k in range(4, 8):
                bbox_car.append(vector_str[k])
            continue
        if ('Pedestrian' == vector_str[0]):
            count_p += 1
            for k in range(4, 8):
                bbox_per.append(vector_str[k])
            continue
        if('Cyclist' == vector_str[0]):
            count_cyc += 1
            for k in range(4, 8):
                bbox_cyc.append(vector_str[k])
            continue

    num = count_c + count_p + count_cyc

    final_line = img_line + ' '+ str(num)
    #car 
    final_line += ' '+ str(count_c)
    for i in bbox_car:
        final_line += ' ' + i 
    #pre
    final_line += ' '+ str(count_p)
    for i in bbox_per:
        final_line += ' ' + i 
    #cyc
    final_line += ' '+ str(count_cyc)
    for i in bbox_cyc:
        final_line += ' ' + i 

    return final_line + '\n' 

def convertKitti(fileIndexList, savedFilename):
    if os.path.exists(fileIndexList):
        file = open(fileIndexList, 'r')
        final_lines_list = []
        for line in file.xreadlines():
            line = line.strip('\n')
            print line
            labelFile = getLabelFilename(line)
            #print labelFile
            finalLine = parse_line(line, labelFile)
            final_lines_list.append(finalLine)
    result_file = open('../data_object_image_2/' + savedFilename, 'w')
    result_file.writelines(final_lines_list)
    result_file.close()


if '__main__' == __name__:
    convertKitti('../data_object_image_2/trainList.txt', 'TrainIndex.txt')




