from collections import defaultdict
import glob
import json
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(
        description='specifications of hyperparameters of cifar-fs-base.')

parser.add_argument('--version_number', type=int, default='1')
parser.add_argument('--random_seed', type=int, default=None)

args = parser.parse_args()

######## repartition classes ##########
if args.random_seed:
    np.random.seed(args.random_seed)

pathname = '/home/oscarli/projects/meta-analysis-classification/data/cifar-fs-base' # original cifar-fs-base dataset path

classes = []

f = open(pathname + '/cifar100/splits/bertinetto/train.txt')
classes.extend([cl.strip() for cl in f.readlines()])
f = open(pathname + '/cifar100/splits/bertinetto/val.txt')
classes.extend([cl.strip() for cl in f.readlines()])
f = open(pathname + '/cifar100/splits/bertinetto/test.txt')
classes.extend([cl.strip() for cl in f.readlines()])

np.random.shuffle(classes)
train_classes = classes[:64]
val_classes = classes[64:80]
test_classes = classes[80:]

########## where to store the new json files ##########
new_root = f'/home/oscarli/projects/meta-analysis-classification/data/cifar-fs-base-repartition-v{args.version_number}-rs{args.random_seed}/'
os.makedirs(new_root)

########## meta-train ##########
base = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
base_test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}


# np.random.seed(seed=11)
count = 0
for each in train_classes:
    each = each.strip()
    indices_base_test = set(np.random.choice(a=600, size=100, replace=False))
    base['label_names'].append(each)
    base_test['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for idx, image_name in enumerate(files):
        if idx in indices_base_test:
            base_test['image_names'].append(image_name)
            base_test['image_labels'].append(count)
        else:
            base['image_names'].append(image_name)
            base['image_labels'].append(count)
    count +=1


json.dump(base , open(new_root + 'base.json','w'))
json.dump(base_test , open(new_root + 'base_test.json','w'))


########## meta-val ##########
val = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

# now count is 64
for each in val_classes:
    each = each.strip()
    val['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for image_name in files:
        val['image_names'].append( image_name)
        val['image_labels'].append(count)
    count +=1


json.dump(val , open(new_root + 'val.json','w'))


########## meta-test ##########
test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

# now count is 80
for each in test_classes:
    each = each.strip()
    test['label_names'].append(each)
    files = sorted(glob.glob( pathname + '/cifar100/data/' + each + '/*'))
    for image_name in files:
        test['image_names'].append( image_name)
        test['image_labels'].append(count)
    count +=1

json.dump(test , open(new_root + 'novel.json','w'))
