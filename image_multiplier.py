#%%
from PIL import Image
import os, sys
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

# %%
# path = 'image/lagavulin_16/*'
path_parent = 'image/*'
path_parent = os.path.dirname(os.path.abspath(path_parent)) + '/'
dirs_parent = os.listdir(path_parent)

for item_parent in dirs_parent:
    print(item_parent)

    path = 'image/' + item_parent + '/*'
    path = os.path.dirname(os.path.abspath(path)) + '/'
    dirs = os.listdir(path)

    # for item in dirs:
    #     if 'png' in item or 'jpg' in item or 'jpeg' in item:
    #         print(item)

    for item in dirs:
        if  (os.path.isfile(path+item)) & ('png' in item or 'jpg' in item or 'jpeg' in item or 'PNG' in item or 'JPG' in item or 'JPEG' in item):
            im = Image.open(path+item)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            #f, e = os.path.splitext(path+item)
            imResize = im.resize((250,250), Image.ANTIALIAS)
            imResize.save(path + item , 'JPEG', quality=90)
            
    fps = glob.glob(path+'*')
    images = np.array(
        [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in fps], 
        dtype = np.uint8
    )

    #=================== Flip, Crop, GaussianBlur, Contrast, Gaussian noise, Ligteness, Affine ( rigbody transform + shear ) ======================================
    seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5, # gaussian blur with random sigma 0~0.5 in half of images
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.ContrastNormalization((0.75, 1.5)),# Strengthen or weaken the contrast in each image.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),    # Add gaussian noise.
    iaa.Multiply((0.8, 1.2), per_channel=0.2), # configure lighteness
    iaa.Affine(    # Affine Transform : Scale/zoom them, translate/move them, rotate them and shear them.
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    ], random_order=True) # apply augmenters in random order


    aug_times = 30

    for times in range(aug_times):
        images_aug = seq(images=images)


    # fig=plt.figure(figsize=(16, 16))
    # columns = 5
    # rows = 2
    # for i in range(1, columns*rows +1):
    #     img = images_aug[i-1]
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()

        i=0
        for img in images_aug:
            cv2.imwrite(os.path.join(path, f'{times}' + 'hauged_'+os.path.basename(fps[i])), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            i += 1

# %%
#=================== Crop, Flip, GaussianBlur only ======================================
# seq = iaa.Sequential([
#     iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#     iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
# ])

# images_aug = seq(images=images)
# print(images_aug.shape)

# # %%
# # fig=plt.figure(figsize=(16, 16))
# # columns = 2
# # rows = 5
# # for i in range(1, columns*rows +1):
# #     img = images_aug[i-1]
# #     fig.add_subplot(rows, columns, i)
# #     plt.imshow(img)
# # plt.show()

# # %%
# def saveSaugImages(path_aug):
#     i=0
#     for img in images_aug:
#         cv2.imwrite(os.path.join(path_aug , 'sauged_'+ os.path.basename(fps[i])), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         i += 1

# saveSaugImages(path)

