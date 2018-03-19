import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from category_tree import category_label
'''
train_file_path='/usa/psu/Documents/CISC849/data/train/'
test_file_path='/usa/psu/Documents/CISC849/data/test/'

max_l_train=0
max_w_train=0
max_l_test=0
max_w_test=0

for file_i in os.listdir(train_file_path):
    data=mpimg.imread(train_file_path+file_i)
    #length, width, channel
    l,w,c=data.shape
    if l>max_l_train:
        max_l_train=l
    if w>max_w_train:
        max_w_train=w

for file_i in os.listdir(test_file_path):
    data=mpimg.imread(test_file_path+file_i)
    #length, width, channel
    l,w,c=data.shape
    if l>max_l_test:
        max_l_test=l
    if w>max_w_test:
        max_w_test=w

#Max length of train: 347
#Max width of train: 430
#Max length of test: 312
#Max width of test: 265

print("Max length of train:",max_l_train)
print("Max width of train:",max_w_train)
print("Max length of test:",max_l_test)
print("Max width of test:",max_w_test)
'''
#win_path='C:\\Users\\schu\\OneDrive\\Course\\Robotics_vision_and_learning\\HW2\\test_data\\device_622.png'
def padding_image(file_path,file_name):
    #ubu_path='/usa/psu/Documents/CISC849/data/test/device_622.png'
    data=mpimg.imread(file_path+file_name)
    #plt.figure()
    #plt.imshow(data)
    #plt.show()
    #print(data.shape)

    l,w,c=data.shape
    new_l=350
    new_w=430
    new_image=np.zeros((new_l,new_w,3))
    length_diff_up=(new_l-l)//2
    width_diff_left=(new_w-w)//2
    #put the image in the center of the new image
    new_image[length_diff_up:length_diff_up+l,width_diff_left:width_diff_left+w,:]=data
    #generate the up part of the new image
    up_edge=data[0:1,0:w,:]

    up_part=up_edge

    for ii in range(length_diff_up-1):
        up_part=np.concatenate((up_part,up_edge),axis=0)
    new_image[0:length_diff_up,width_diff_left:width_diff_left+w,:]=up_part

    #generate the bottom part of the new image
    bottom_edge=data[l-1:l,0:w,:]
    bottom_part=bottom_edge

    for ii in range(length_diff_up-1):
        bottom_part=np.concatenate((bottom_part,bottom_edge),axis=0)
    new_image[length_diff_up+l:length_diff_up*2+l,width_diff_left:width_diff_left+w,:]=bottom_part

    #generate the left part of the image
    left_edge=new_image[0:new_l,width_diff_left:width_diff_left+1,:]

    left_part=left_edge
    for ii in range(width_diff_left-1):
        left_part=np.concatenate((left_part,left_edge),axis=1)

    new_image[0:new_l,0:width_diff_left,:]=left_part

    #generate the right part of the image
    right_edge=new_image[0:new_l,width_diff_left+w-1:width_diff_left+w,:]

    right_part=right_edge
    for ii in range(width_diff_left-1):
        right_part=np.concatenate((right_part,right_edge),axis=1)

    new_image[0:new_l,width_diff_left+w:width_diff_left*2+w,:]=right_part
    #plt.figure()
    #plt.imshow(new_image)
    #plt.show()
    #get the label of the image
    #fruit [1,0,0,0]
    #device [0,1,0,0]
    #vegetable [0,0,1,0]
    #container [0,0,0,1]
    file_label=file_name.split('_')
    label=category_label[file_label[0]]
    #training_data=[[new_image],[[1,0,0,0]]]

    return new_image,label


def down_sample_image(original_image,pixel_interval):
    return original_image[::pixel_interval,::pixel_interval,:]



def iter_dataset(file_path,model,batch_size,down_sample=False,pixel_interval=1):
    final_path=file_path+model+'/'
    training_data=[[],[]]
    file_num=0
    for file_i in os.listdir(final_path):
        image,label=padding_image(final_path,file_i)
        if down_sample is True:
            ds_image=down_sample_image(image,pixel_interval)
        else:
            ds_image=image
        training_data[0].append(ds_image)
        training_data[1].append(label)
        file_num+=1
        if file_num==batch_size:
            yield training_data
            training_data=[[],[]]
            file_num=0
    if file_num>0:
        yield training_data




if __name__ == '__main__':
    file_path='/usa/psu/Documents/CISC849/test_image//'
    model='test'
    for ii in iter_dataset(file_path,model,10,True,4):
        print(np.array(ii[0]).shape)