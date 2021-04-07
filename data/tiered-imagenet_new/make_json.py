import glob
import json
import os
import numpy as np

pathname = os.getcwd()
print(pathname)


########## meta-train ##########
base = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

np.random.seed(seed=12)
count = 0
for cl in sorted(os.listdir(os.path.join(pathname, 'train'))):
    cl_folder = os.path.join(pathname, 'train', cl)
    base['label_names'].append(cl)
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    for image_name in files:
        base['image_names'].append(image_name)
        base['image_labels'].append(count)
    count +=1


json.dump(base , open('base.json','w'))


########## meta-val ##########
val = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

for cl in sorted(os.listdir(os.path.join(pathname, 'val'))):
    cl_folder = os.path.join(pathname, 'val', cl)
    val['label_names'].append(cl)
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    for image_name in files:
        val['image_names'].append(image_name)
        val['image_labels'].append(count)
    count +=1

json.dump(val , open('val.json','w'))


########## meta-test ##########
test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

for cl in sorted(os.listdir(os.path.join(pathname, 'test'))):
    cl_folder = os.path.join(pathname, 'test', cl)
    test['label_names'].append(cl)
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    for image_name in files:
        test['image_names'].append(image_name)
        test['image_labels'].append(count)
    count += 1

json.dump(test , open('novel.json','w'))
