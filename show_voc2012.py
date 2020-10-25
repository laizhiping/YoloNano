import torchvision.datasets as datasets
import torchvision
import torch
import numpy as np
import cv2
from PIL import Image

def show_object_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0,255,255), 2)


def show_object_name(image: np.ndarray, name: str, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 0, 0))
     
'''
CLASS torchvision.datasets.VOCDetection(root, year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=None)[SOURCE]
Pascal VOC Detection Dataset.

Parameters
root (string) – Root directory of the VOC Dataset.

year (string, optional) – The dataset year, supports years 2007 to 2012.

image_set (string, optional) – Select the image_set to use, train, trainval or val

download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. (default: alphabetic indexing of VOC’s 20 classes).

transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop

target_transform (callable, required) – A function/transform that takes in the target and transforms it.

transforms (callable, optional) – A function/transform that takes input sample and its target as entry and returns a transformed version.
'''   
voc_trainset = datasets.VOCDetection('./Dataset/VOCtrainval_11-May-2012',year='2012', image_set='train', download=False)
# print(voc_trainset)
# print('-'*40)
# print('VOC2012-trainval')
# print(len(voc_trainset))

for i, sample in enumerate(voc_trainset, 1):
    # print(sample)
    image, annotation = sample[0], sample[1]['annotation']
    img_w = int(annotation['size']['width'])
    img_h = int(annotation['size']['height'])
    image = image.resize((416, 416), Image.ANTIALIAS)
    objects = annotation['object']
    show_image = np.array(image)
    # print('{} object:{}'.format(i, len(objects)))
    if not isinstance(objects,list):
        object_name = objects['name']
        object_bndbox = objects['bndbox']
        x_min = int(object_bndbox['xmin']) 
        y_min = int(object_bndbox['ymin'])
        x_max = int(object_bndbox['xmax'])
        y_max = int(object_bndbox['ymax'])

        x_min = int(int(object_bndbox['xmin']) * 416 / img_w) 
        y_min = int(int(object_bndbox['ymin']) * 416 / img_h)
        x_max = int(int(object_bndbox['xmax']) * 416 / img_w)
        y_max = int(int(object_bndbox['ymax']) * 416 / img_h)
        show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
        show_image =show_object_name(show_image, object_name, (x_min, y_min))
    else:
        for j in objects:
            object_name = j['name']
            object_bndbox = j['bndbox']
            x_min = int(object_bndbox['xmin']) 
            y_min = int(object_bndbox['ymin'])
            x_max = int(object_bndbox['xmax'])
            y_max = int(object_bndbox['ymax'])

            x_min = int(int(object_bndbox['xmin']) * 416 / img_w) 
            y_min = int(int(object_bndbox['ymin']) * 416 / img_h)
            x_max = int(int(object_bndbox['xmax']) * 416 / img_w)
            y_max = int(int(object_bndbox['ymax']) * 416 / img_h)
            # print(x_min, y_min, x_max, y_max)
            show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
            show_image = show_object_name(show_image, object_name, (x_min, y_min))

    cv2.imshow('image', show_image)
    cv2.waitKey(0)


# print(voc_trainset)
# print('Down load ok')