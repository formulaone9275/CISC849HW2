from category_tree import category_tree,category
import os
import shutil
file_path='/usa/psu/Downloads/rgbd-dataset/'
destination_path='/usa/psu/Documents/CISC849/depth_data'
#build the training dataset
training_file_list=[]
test_file_list=[]
category_num_train={}
category_num_test={}
for c in category:
    category_num_test[c]=0
    category_num_train[c]=0
    for ct_i in category_tree[c]:
        for sub_folder_i in os.listdir(file_path+ct_i):
            file_instance_path=os.path.join(file_path+ct_i,sub_folder_i)
            #print(file_instance_path+'/')
            #print(os.listdir(file_instance_path))
            #get the depthcrop images
            crop_file_list=[]
            for file_i in os.listdir(file_instance_path):
                if '_depthcrop' in file_i:
                    crop_file_list.append(file_i)
            #down sampling
            for image_i in crop_file_list:
                file_name_split=image_i.split('_')
                #get the frame number
                frame_sequence_num=file_name_split[len(file_name_split)-2]
                #get the instance number
                instance_num=file_name_split[len(file_name_split)-4]
                '''
                #get the object name
                object_name_list=file_name_split[0:len(file_name_split)-4]
                if len(object_name_list)==1:
                    object_name=object_name_list[0]
                elif len(object_name_list)==2:
                    object_name=object_name_list[0]+'_'+object_name_list[1]
                '''
                if int(frame_sequence_num)%5==0:
                    if int(instance_num)==1:
                        #for testing
                        test_file_list.append(image_i)
                        test_name=c+'_'+str(category_num_test[c])+'.png'
                        print('test')
                        print(file_instance_path+'/'+image_i,' ',destination_path+'/'+'test/'+test_name)
                        shutil.copy(file_instance_path+'/'+image_i,destination_path+'/'+'test/'+test_name)
                        category_num_test[c]+=1
                    else:
                        #for training
                        training_file_list.append(image_i)
                        train_name=c+'_'+str(category_num_train[c])+'.png'
                        print('train')
                        print(file_instance_path+'/'+image_i,' ',destination_path+'/'+'train/'+train_name)
                        shutil.copy(file_instance_path+'/'+image_i,destination_path+'/'+'train/'+train_name)
                        category_num_train[c]+=1





