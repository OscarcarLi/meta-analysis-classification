import os
from collections import defaultdict
import random
import json

# change this to the path of the folder containing three files:
# identity_CelebA.txt  img_align_celeba  list_attr_celeba.txt
# img_align_celeba is the unzipped folder from img_align_celeba.zip (containing jpeg files)
celeba_root = os.getcwd()

with open(os.path.join(celeba_root, 'list_attr_celeba.txt'), 'r') as file:
    file_iter = iter(file)
    _ = next(file_iter) # throw away the number of images 202599
    attribute_names = next(file_iter)
    # get the attribute name and construct a mapping from the attribute name to a unique location index
    attribute_to_index = {attribute: i for i, attribute in enumerate(attribute_names.split())}
    
    image_to_attribute = {}
    # every remaining line is the name and attribute values for a specific image
    for line in file_iter:
        sample = line.split()
        if len(sample) != 41:
            raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
        image_name = sample[0]
        image_to_attribute[image_name] = [int(i) for i in sample[1:]]

print(f'the total number of images is {len(image_to_attribute.keys())}')

# define a labelling function based on the attributes of interest
labelling_function = lambda x: int((x[attribute_to_index['Smiling']] + 1) / 2)

print(f"the label for image '000019.jpg' is {labelling_function(image_to_attribute['000019.jpg'])}")

client_to_class_to_imagepathlist = defaultdict(lambda: defaultdict(list))
with open(os.path.join(celeba_root, 'identity_CelebA.txt'), 'r') as file:
    for line in file:
        # every is an image and the corresponding celebrity's id
        image_name, celeb_id = line.split()
        label = labelling_function(image_to_attribute[image_name])
        full_image_path = os.path.join(celeba_root, 'img_align_celeba', image_name)
        client_to_class_to_imagepathlist[celeb_id][label].append(full_image_path)

print(f"client example list: {client_to_class_to_imagepathlist['2880']}")

min_num_examples_per_class = 3
total_client_list = []
all_cl_list = [0, 1] # this depends on the labelling function and should be changed accordingly
for client_id in client_to_class_to_imagepathlist.keys():
    have_enough_examples = True
    for cl in all_cl_list: # the client needs to have at least min_num_examples_per_class for every single class
        # the client cannot have missing class
        if len(client_to_class_to_imagepathlist[client_id][cl]) < min_num_examples_per_class:
            have_enough_examples = False
            break
    if have_enough_examples:
        total_client_list.append(client_id)

print(f'the total number of clients with at least {min_num_examples_per_class} examples for each class is {len(total_client_list)}')
total_client_list = sorted(total_client_list)

random.seed(a=42)
random.shuffle(total_client_list)

num_train = int(len(total_client_list) * 0.6)
num_val = int(len(total_client_list) * 0.2)

base_client_list = total_client_list[:num_train]
val_client_list = total_client_list[num_train:num_train + num_val]
novel_client_list = total_client_list[num_train + num_val:]

json_name_to_client_list = {
    'base.json': base_client_list,
    'val.json': val_client_list,
    'novel.json': novel_client_list,
}

print(f'meta-train client number {len(base_client_list)}')
print(f'meta-val client number {len(val_client_list)}')
print(f'meta-test client number {len(novel_client_list)}')

for json_name, client_list in json_name_to_client_list.items():
    with open(json_name, 'w') as f:
        json.dump(obj={client_id: client_to_class_to_imagepathlist[client_id] for client_id in client_list},
                  fp=f)
        print(f'save {json_name}')
