import sys
import os
from random import sample
from tqdm import tqdm
import numpy as np
import json
import argparse
from PIL import Image
import random

def main(args):
    dataset_path = os.path.join('dataset', args.dataset_name)
    if not os.path.exists(dataset_path):
        raise Exception('dataset path % s does not exist' % dataset_path)
    all_model_names = next(os.walk(dataset_path))[1]
    counter = 0
    dataset_catalog = {}
    for model in all_model_names:
        dir_temp = os.path.join(dataset_path, model, 'images')
        all_images =  os.listdir(dir_temp)
        for image_name in all_images:
            image_path = os.path.join(model, 'images', image_name)
            label_path = os.path.join(model, 'labels', image_name[:-4]+'_label.png')
            dataset_catalog[counter] = {'image': image_path, 'label':label_path}
            counter +=1
            
    texture_catalog = {}
    texture_path = "textures/random"
    destination_path = os.path.join('dataset', args.dataset_name)

    for texture_name in os.listdir(texture_path):
        # image_path = os.path.join(texture_path, texture)
        texture_catalog[counter] = texture_name
        counter +=1
        
    for _, var in tqdm(dataset_catalog.items()):
        image = np.array(Image.open(os.path.join(dataset_path, var['image'])))
        label = Image.open(os.path.join(dataset_path, var['label']))
        label_np = np.array(label)
        rand_key = random.choice(texture_catalog.keys())
        texture = Image.open(os.path.join(texture_path, texture_catalog[rand_key]))
        width, height = image.shape[1], image.shape[0]
        texture = np.array(texture.resize((width, height)))
        bg = (label_np == 0)
        image[bg] = texture[bg]
        #
        image_dr = Image.fromarray(np.uint8(image))
        image_path = os.path.join(destination_path, var['image'])
        image_dir = os.path.dirname(image_path)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_dr.save(image_path)
        #
        label_path = os.path.join(destination_path, var['label'])
        label_dir = os.path.dirname(label_path)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        # os.makedirs(label_path)
        label.save(label_path)
        

if __name__ == '__main__':
    generator_parser = argparse.ArgumentParser(
        description='Split dataset for trainig and evaluation')
    generator_parser.add_argument('--dataset_name', type=str, default='',
                                  help='dataset name')   
    args = generator_parser.parse_args()
    
    main(args)

