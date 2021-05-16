import json
import os
import shutil
import tqdm

json_path = '/usr2/asetlur/meta-learning/meta-analysis-classification/data/miniImagenet-base-nl552' # the folder containing the current jsons
# the goal is to change the content of the current json

name_map = {'novel_large.json': 'test_large552',}

for key in name_map.keys():
    with open(os.path.join(json_path, key), 'r') as file:
        json_object = json.load(file)
        
    new_file_path_list = [] # the new list of file paths

    for file_path in tqdm.tqdm(json_object['image_names']):

        idx2 = file_path.rfind('/')
        idx = file_path.rfind('/', 0, idx2) + 1

        dst = os.path.join(json_path, name_map[key], file_path[idx:idx2]) # file_path[idx:idx2] is the folder name

        new_file_path_list.append(os.path.join(dst, file_path[idx2 + 1:])) # file_path[idx2 + 1 : ] is the file name
        
    json_object['image_names'] = new_file_path_list # update the file paths

    with open(os.path.join(json_path, key), 'w') as file:
        json.dump(json_object, file)
