#import pydicom as dicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itk
# import pylibjpeg-libjpeg
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import math
import pickle
import pydicom as dicom

#heetha
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import random

file_path="/home/thuvaaragan/FYP/kaggle_2/train_images/" #image .dcm folder path
csv_path ="/home/thuvaaragan/FYP/kaggle_2/df_Positive.csv"  # path to csv
# positive_path = "C:/Users/Nirho/Desktop/positive/"  #path to postive cases folder
# negative_path = "C:/Users/Nirho/Desktop/negative/"  #path to negative cases folder

def patch_img(ds,side):
    pixel_array_numpy = np.asarray(ds)
    shape=pixel_array_numpy.shape
    pixel_array_numpy = pixel_array_numpy.reshape(1, *shape, 1)
    image = tf.convert_to_tensor(pixel_array_numpy)
    a= tf.image.extract_patches(images=image, sizes=[1, 694, 520, 1], strides=[1, 694, 520, 1], rates=[1, 1, 1, 1], padding='VALID')
    patch_list = []
    sub_list = []
    if side == "R":
        for i in range(math.floor(shape[0]/694)):
            for j in range(math.floor(shape[1]/520)):
                patch =tf.reshape(a[0,i,j,:],[694,520])
                patch = np.asarray(patch)
                T=np.sum(patch)/(694*520)
                if T >=300: # tresh hold
                    patch_image = Image.fromarray(patch)
                    zoom_patch = patch_image.resize((1388,1040))
                    patch_list.append([i,j,zoom_patch])
                elif(i==3 and j==3) or (i==3 and j==4):
                    patch_image = Image.fromarray(patch)
                    zoom_patch = patch_image.resize((1388,1040))
                    sub_list.append([i,j,zoom_patch])

    elif side == "L":
        for i in range(math.floor(shape[0]/694)):
            for j in range(math.floor(shape[1]/520)):
                patch =tf.reshape(a[0,i,j,:],[694,520])
                patch = np.asarray(patch)
                T=np.sum(patch)/(694*520)
                if T >=300: # tresh hold
                    patch_image = Image.fromarray(patch)
                    zoom_patch = patch_image.resize((1388,1040))
                    patch_list.append([i,j,zoom_patch]) # patch location , patch
                elif(i==math.floor(shape[0]/694)-3 and j==math.floor(shape[1]/520)-3) or (i==math.floor(shape[0]/694)-3 and j==math.floor(shape[1]/520)-4):
                    patch_image = Image.fromarray(patch)
                    zoom_patch = patch_image.resize((1388,1040))
                    sub_list.append([i,j,zoom_patch])
    # return patch_list

    if len(patch_list)>=2:
         return patch_list
    else:
        return sub_list


def image_feature(patch_list):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    c=0
    for i in patch_list:
        reshape_pa = i[2].resize((224,224)) 
        pa_image = np.stack((np.asarray(reshape_pa),)*3, axis=-1)
        x = pa_image
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(c)
        c=c+1
    return features,img_name

def get_2_patches(patch):
    patch_features,patch_ids=image_feature(patch)
    patch_cluster = pd.DataFrame(patch_ids,columns=['patch'])
    k = 2
    clusters = KMeans(k, random_state = 40)
    clusters.fit(patch_features)
    patch_cluster["clusterid"] = clusters.labels_
    patch_cluster[patch_cluster.clusterid == 0]
    cl_0=[]
    cl_1=[]
    for i in range(len(patch_cluster)):
        if patch_cluster['clusterid'][i]==0:
            cl_0.append(patch[i])
        else:
            cl_1.append(patch[i])
    return random.choice(cl_0),random.choice(cl_1)


def patch_from_dcm(ds,side):
    a = []
    patch_list=patch_img(ds,side)
    c0,c1 = get_2_patches(patch_list)
    a.append(c0)
    a.append(c1)
    return a

def main():
    dict_cc={}
    dict_mlo={}
    key1=['image_array','label']
    dict_cc[key1[0]]=[]
    dict_cc[key1[1]] = []
    dict_mlo[key1[0]] = []

    dict_mlo[key1[1]] = []
    print(dict_mlo.keys())
    meta_data=pd.read_csv(csv_path,header=None)
    meta_data=np.array(meta_data)
    #print(meta_data)
    print(meta_data.shape)
    print(meta_data[1][1],meta_data[1][2],meta_data[1][4],meta_data[1][4],meta_data[1][6])

    # meta_data.shape[0]
    for i in range(450,454):
        print(i)
        # if i==10:
        #     break

        image_path=file_path+str(meta_data[i][1])+"/"+str(meta_data[i][2])+".dcm"
        img=itk.imread(image_path)
        # img=dicom.dcmread(image_path)

        # print(img.shape)
        patch = patch_from_dcm(img[0], str(meta_data[i][3]))
        # print(len(patch))
        # patch = patch_img(img, str(meta_data[i][3]))
        if str(meta_data[i][4])=="CC":
            for k in range(len(patch)):
                dict_cc[key1[0]].append(patch[k])
                dict_cc[key1[1]].append(meta_data[i][6])
        elif str(meta_data[i][4])=="MLO":
            for k in range(len(patch)):
                dict_mlo[key1[0]].append(patch[k])
                dict_mlo[key1[1]].append(meta_data[i][6])
        #print(meta_data[i][2])
    #print("CC")
    #print(dict_cc)
    #print("MLO")
    #print(dict_mlo)
    print("save file")
    save_file_cc = '/home/thuvaaragan/FYP/kaggle_2/Image_Array_all/cc_positive_11.pkl'
    save_file_mlo = '/home/thuvaaragan/FYP/kaggle_2/Image_Array_all/mlo_positive_11.pkl'
    outfile_cc = open(save_file_cc, 'wb')
    pickle.dump(dict_cc, outfile_cc)
    outfile_cc.close()
    # if not os.path.exists(save_file_cc):
    #     print('file not saved')
    outfile_mlo = open(save_file_mlo, 'wb')
    pickle.dump(dict_mlo, outfile_mlo)
    outfile_mlo.close()

if __name__ == main():
    main()

#negative
# 5 - 440,442
#postive6 500-599
#postive7 600-652
#postive8 654-682
#postive9 684-701
#postive10 703-747
#postive11 448-452
