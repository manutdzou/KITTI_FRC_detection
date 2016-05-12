clear all;close all;clc;
image_list=importdata('ImageList_Version_S_AddData.txt','r');
rand('seed',2);
index=randperm(length(image_list));
fp_train=fopen('ImageList_Version_S_AddData_train.txt','w');
fp_val=fopen('ImageList_Version_S_AddData_val.txt','w');
fp_train_gt=fopen('ImageList_Version_S_GT_AddData_train.txt','w');
fp_val_gt=fopen('ImageList_Version_S_GT_AddData_val.txt','w');
trainsample=0.7;% 70% as training and 30% validation
train_num=fix(trainsample*length(index));
train_index=index(1:train_num);
val_index=index(train_num+1:end);

fidin1=fopen('ImageList_Version_S_AddData.txt','r');
fidin2=fopen('ImageList_Version_S_GT_AddData.txt','r');

ind=1;
while ~feof(fidin1) 
    tline=fgetl(fidin1);
    if length(find(train_index==ind))==1
        fprintf(fp_train,tline);
        fprintf(fp_train,'\n');
    else
        fprintf(fp_val,tline);
        fprintf(fp_val,'\n');
    end
    ind=ind+1;
end

ind=1;
while ~feof(fidin2) 
    tline=fgetl(fidin2);
    if length(find(train_index==ind))==1
        fprintf(fp_train_gt,tline);
        fprintf(fp_train_gt,'\n');
    else
        fprintf(fp_val_gt,tline);
        fprintf(fp_val_gt,'\n');
    end
    ind=ind+1;
end
