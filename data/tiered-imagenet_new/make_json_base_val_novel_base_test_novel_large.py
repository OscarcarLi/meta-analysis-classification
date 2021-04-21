import glob
import json
import os
import numpy as np

pathname = os.getcwd()
print(pathname)


########## meta-train ##########
base = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
base_test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

np.random.seed(seed=12)
count = 0
for cl in sorted(os.listdir(os.path.join(pathname, 'train'))):
    cl_folder = os.path.join(pathname, 'train', cl)
    num_examples = len(list(os.listdir(cl_folder)))
    indices_base_test = set(np.random.choice(a=num_examples, size=int(0.2 * num_examples), replace=False))
    base['label_names'].append(cl)
    base_test['label_names'].append(cl)
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    for idx, image_name in enumerate(files):
        if idx in indices_base_test:
            base_test['image_names'].append(image_name)
            base_test['image_labels'].append(count)
        else:
            base['image_names'].append(image_name)
            base['image_labels'].append(count)
    count +=1


json.dump(base , open('base.json','w'))
json.dump(base_test , open('base_test.json','w'))


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
test_synset_to_int_label = {}

for cl in sorted(os.listdir(os.path.join(pathname, 'test'))):
    cl_folder = os.path.join(pathname, 'test', cl)
    test['label_names'].append(cl)
    test_synset_to_int_label[cl] = count # record the test class's unique integer label
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    for image_name in files:
        test['image_names'].append(image_name)
        test['image_labels'].append(count)
    count += 1

json.dump(test , open('novel.json','w'))


########## meta-test ##########
test_large = {'label_names': [] , 'image_names':[] , 'image_labels':[]}

for cl in sorted(os.listdir(os.path.join(pathname, 'test_large'))):
    cl_folder = os.path.join(pathname, 'test_large', cl)
    test_large['label_names'].append(cl)
    files = sorted(glob.glob(os.path.join(cl_folder, '*')))
    if cl in test_synset_to_int_label:
        # this means the class synset cl has appeared in the test set
        # so we use the same unique integer label used in the folder test.
        label = test_synset_to_int_label[cl]
    else:
        # we haven't used this class in the test set, so we assign a unique integer label to it
        label = count
        count += 1

    for image_name in files:
        test_large['image_names'].append(image_name)
        test_large['image_labels'].append(label)

json.dump(test_large , open('novel_large.json','w'))
