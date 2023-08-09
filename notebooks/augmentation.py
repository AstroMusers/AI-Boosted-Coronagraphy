import random
import PIL
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
import os
from tqdm import tqdm
import glob



def horizontal_flip(image):
    horizontal_flip = A.HorizontalFlip(p=0.99)
    augmented_image = horizontal_flip(image=image)['image']

    return augmented_image

def vertical_flip(image):
    vertical_flip   = A.VerticalFlip(p=0.99)
    augmented_image = vertical_flip(image=image)['image']

    return augmented_image

def rotate(image):
    random_rotate   = A.Rotate(limit=355,always_apply=True)
    augmented_image = random_rotate(image=image)['image']

    return augmented_image

def shift(image):
    shift = A.ShiftScaleRotate(shift_limit=0.235, scale_limit=0, rotate_limit=0, p=1)
    augmented_image = shift(image=image)['image']

    return augmented_image


def save_image(image, save_name, proposal_id, augmentations):
    DIR = f'/data/scratch/bariskurtkaya/dataset/NIRCAM/{proposal_id}/sci_imgs'
    im = PIL.Image.fromarray(image)
    im = im.convert("L")
    im.save(os.path.join(DIR, save_name + f'_{augmentations}.jpg'))
    

def augment(img_dirs, proposal_id):

    for img in tqdm(img_dirs):
        image   = PIL.Image.open(img)
        image   = np.array(image)
        save_name = img.split('/')[-1][:-4]

        ## horizontal flip
        hor_img = horizontal_flip(image)
        aug = 'horflip'
        save_image(hor_img, save_name, proposal_id, aug)

        ## vertical flip 
        ver_img = vertical_flip(image)
        aug = 'verflip'
        save_image(ver_img, save_name, proposal_id, aug)

        ## rotate
        rot_img = rotate(image)
        aug = 'rot'
        save_image(rot_img, save_name, proposal_id, aug)

        ## shift
        shift_img = shift(image)
        aug = 'shift'
        save_image(shift_img, save_name, proposal_id, aug)

        #print('First stage ends')


        ## horizontal + vertical
        hor_ver_img = vertical_flip(hor_img)
        aug = 'horflip_verflip'
        save_image(hor_ver_img, save_name, proposal_id, aug)

        ## horizontal + rotate
        hor_rot_img = rotate(hor_img)
        aug = 'horflip_rot'
        save_image(hor_rot_img, save_name, proposal_id, aug)

        ## horizontal + shift
        hor_shift_img = shift(hor_img)
        aug = 'horflip_shift'
        save_image(hor_shift_img, save_name, proposal_id, aug)



        ## vertical + horizontal
        ver_hor_img = horizontal_flip(ver_img)
        aug = 'verflip_horflip'
        save_image(ver_hor_img, save_name, proposal_id, aug)

        ## vertical + rotate
        ver_rot_img = rotate(ver_img)
        aug = 'verflip_rot'
        save_image(ver_rot_img, save_name, proposal_id, aug)

        ##vertical + shift
        ver_shift_img = shift(ver_img)
        aug = 'verflip_shift'
        save_image(ver_shift_img, save_name, proposal_id, aug)



        ## rotate + vertical
        rot_ver_img = vertical_flip(rot_img)
        aug = 'rot_verflip'
        save_image(rot_ver_img, save_name, proposal_id, aug)

        ## rotate + horizontal
        rot_hor_img = horizontal_flip(rot_img)
        aug = 'rot_horflip'
        save_image(rot_hor_img, save_name, proposal_id, aug)

        ## rotate + shift
        rot_shift_img = shift(rot_img)
        aug = 'rot_shift'
        save_image(rot_shift_img, save_name, proposal_id, aug)



        ## shift + vertical
        shift_ver_img = vertical_flip(shift_img)
        aug = 'shift_verflip'
        save_image(shift_ver_img, save_name, proposal_id, aug)

        ## shift + horizontal
        shift_hor_img = horizontal_flip(shift_img)
        aug = 'shift_horflip'
        save_image(shift_hor_img, save_name, proposal_id, aug)

        ## shift + rotate
        shift_rot_img = rotate(shift_img)
        aug = 'shift_rot'
        save_image(shift_rot_img, save_name, proposal_id, aug)

        #print('Second stage ends')


        ## horizontal + vertical + rotate
        hor_ver_rot_img = rotate(hor_ver_img)
        aug = 'horflip_verflip_rot'
        save_image(hor_ver_rot_img, save_name, proposal_id, aug)

        ## horizontal + vertical + shift
        hor_ver_shift_img = shift(hor_ver_img)
        aug = 'horflip_verflip_shift'
        save_image(hor_ver_shift_img, save_name, proposal_id, aug)


        ## horizontal + rotate + vertical
        hor_rot_ver_img = vertical_flip(hor_rot_img)
        aug = 'horflip_rot_verflip'
        save_image(hor_rot_ver_img, save_name, proposal_id, aug)

        ## horizontal + rotate + shift
        hor_rot_shift_img = shift(hor_rot_img)
        aug = 'horflip_rot_shift'
        save_image(hor_rot_shift_img, save_name, proposal_id, aug)


        ## horizontal + shift + rotate
        hor_shift_rotate_img = rotate(hor_shift_img)
        aug = 'horflip_shift_rot'
        save_image(hor_shift_rotate_img, save_name, proposal_id, aug)

        ## horizontal + shift + vertical
        hor_shift_ver_img = vertical_flip(hor_shift_img)
        aug = 'horflip_shift_verflip'
        save_image(hor_shift_ver_img, save_name, proposal_id, aug)




        ## vertical + horizontal + rotate
        ver_hor_rotate_img = rotate(ver_hor_img)
        aug = 'verflip_horflip_rot'
        save_image(ver_hor_rotate_img, save_name, proposal_id, aug)

        ## vertical + horizontal + shift
        ver_hor_shift_img = shift(ver_hor_img)
        aug = 'verflip_horflip_shift'
        save_image(ver_hor_shift_img, save_name, proposal_id, aug)


        ## vertical + rotate + horizontal
        ver_rot_hor_img = horizontal_flip(ver_rot_img)
        aug = 'verflip_rot_horflip'
        save_image(ver_rot_hor_img, save_name, proposal_id, aug)

        ## vertical + rotate + shift
        ver_rot_shift_img = horizontal_flip(ver_rot_img)
        aug = 'verflip_rot_shift'
        save_image(ver_rot_shift_img, save_name, proposal_id, aug)


        ## vertical + shift + horizontal
        ver_shift_hor_img = horizontal_flip(ver_shift_img)
        aug = 'verflip_shift_horflip'
        save_image(ver_shift_hor_img, save_name, proposal_id, aug)

        ## vertical + shift + rotate
        ver_shift_rot_img = rotate(ver_shift_img)
        aug = 'verflip_shift_rotate'
        save_image(ver_shift_rot_img, save_name, proposal_id, aug)



        ## rotate + vertical + horizontal
        rot_ver_hor_img = horizontal_flip(rot_ver_img)
        aug = 'rot_verflip_horflip'
        save_image(rot_ver_hor_img, save_name, proposal_id, aug)

        ## rotate + vertical + shift
        rot_ver_shift_img = shift(rot_ver_img)
        aug = 'rot_verflip_shift'
        save_image(rot_ver_shift_img, save_name, proposal_id, aug)


        ## rotate + horizontal + vertical
        rot_hor_ver_img = vertical_flip(rot_hor_img)
        aug = 'rot_horflip_verflip'
        save_image(rot_hor_ver_img, save_name, proposal_id, aug)

        ## rotate + horizontal + shift
        rot_hor_shift_img = shift(rot_hor_img)
        aug = 'rot_horflip_shift'
        save_image(rot_hor_shift_img, save_name, proposal_id, aug)


        ## rotate + shift + vertical
        rot_shift_ver_img = vertical_flip(rot_shift_img)
        aug = 'rot_shift_verflip'
        save_image(rot_shift_ver_img, save_name, proposal_id, aug)

        ## rotate + shift + horizontal
        rot_shift_hor_img = horizontal_flip(rot_shift_img)
        aug = 'rot_shift_horflip'
        save_image(rot_shift_hor_img, save_name, proposal_id, aug)


        ## shift + vertical + horizontal
        shift_ver_hor_img = horizontal_flip(shift_ver_img)
        aug = 'shift_verflip_horflip'
        save_image(shift_ver_hor_img, save_name, proposal_id, aug)

        ## shift + vertical + rotate
        shift_ver_rot_img = rotate(shift_ver_img)
        aug = 'shift_verflip_rot'
        save_image(shift_ver_rot_img, save_name, proposal_id, aug)


        ## shift + horizontal + vertical
        shift_hor_ver_img = vertical_flip(shift_hor_img)
        aug = 'shift_horflip_verflip'
        save_image(shift_hor_ver_img, save_name, proposal_id, aug)

        ## shift + horizontal + rotate
        shift_hor_rot_img = rotate(shift_hor_img)
        aug = 'shift_horflip_rot'
        save_image(shift_hor_rot_img, save_name, proposal_id, aug)


        ## shift + rotate + vertical
        shift_rot_ver_img = vertical_flip(shift_rot_img)
        aug = 'shift_rot_ver'
        save_image(shift_rot_ver_img, save_name, proposal_id, aug)

        ## shift + rotate + horizontal
        shift_rot_hor_img = horizontal_flip(shift_rot_img)
        aug = 'shift_rot_hor'
        save_image(shift_rot_hor_img, save_name, proposal_id, aug)

        #print('Third stage ends')

img_dirs = sorted(glob.glob('/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/sci_imgs/*'))
print(len(img_dirs))
augment(img_dirs=img_dirs, proposal_id='1386')