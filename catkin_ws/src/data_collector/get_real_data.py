from PIL import Image
from urllib.request import urlopen
import json
import numpy as np
import csv
import os

def get_real_dataset(json_f, output_folder):
    data_dir = os.path.join('dataset', output_folder)
    img_path = os.path.join(data_dir, 'images')
    label_path = os.path.join(data_dir, 'labels')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(img_path)
        os.makedirs(label_path)
    model_f = open('model_list.csv', 'r')
    reader = list(csv.DictReader(model_f))
    model_list = {}
    for _, l in enumerate(reader):
        model_list[l['name']] = int(int(l['label']))
    print(model_list)
    imdb = json.load(open(json_f))
    counter = 0
    for sample in imdb:
        print(f'downloading sample {counter}')
        img_url = sample['Labeled Data']
        img = Image.open(urlopen(img_url))
        instances = sample['Label']['objects']
        w, h = img.size
        label = np.zeros((h, w))
        for item in instances:
            instance_url = item['instanceURI']
            # print(instance_url)
            instance_class = item['value']
            class_label = model_list[instance_class]
            try:
                instance_label = np.array(Image.open(urlopen(instance_url))).astype(np.uint8)
                instance_label = np.sum(instance_label, axis=2)
                instance_label = np.where(instance_label>0, class_label, 0)
                label += instance_label.astype(np.uint8)
            except:
                pass
        counter += 1
        label = Image.fromarray(label.astype(np.uint8))
        print(np.unique(label))
        img.save(os.path.join(img_path, f'{counter}.jpg'))
        label.save(os.path.join(label_path, f'{counter}_label.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get anotated dataset')
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--dataset_name' , type=str, default='')
    args = parser.parse_args()
    get_real_dataset(args.json_file, args.dataset_name)
