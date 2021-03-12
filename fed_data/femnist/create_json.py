import json
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

def get_class_images(class_root_path, user_to_class_to_imagepath):
    """
    Reads image paths correspeonding to a class
    class_root_path is root of class directory
    user_to_class_to_imagepath is a dictionary mapping user->class->imagepath
    """

    class_id = os.path.basename(class_root_path)
    print(f"Reading class {class_id}")
            
    for hsf_fname in os.listdir(class_root_path):
        # read mit files which contain metadata
        if 'mit' in hsf_fname:
            with open(os.path.join(class_root_path, hsf_fname)) as f:
                class_images_details = list(map(
                    lambda x: x.strip().split(), f.readlines()))
            
            # drop first line of class_images_details, since it only contains count
            count, class_images_details = int(class_images_details[0][0]), class_images_details[1:]

            # iterate over class_images_details
            for image_fname, user_info in class_images_details:
                user_id = user_info.split("/")[0]
                
                # add root directory and hsf directory to image_fname
                full_image_path = os.path.join(
                    class_root_path,
                    hsf_fname.split(".")[0], # remove mit extension
                    image_fname
                ) 

                # if user is not present in dict create new user
                if user_id not in user_to_class_to_imagepath: 
                    user_to_class_to_imagepath[user_id] = defaultdict(list)
                
                # add data to class/user dict
                user_to_class_to_imagepath[user_id][class_id].append(full_image_path)

    return user_to_class_to_imagepath
                
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--femnist-root', type=str, required=True)
    args = parser.parse_args()

    user_to_class_to_imagepath = {}
    classes_home = os.path.join(args.femnist_root, "data/raw_data/by_class")
    for class_name in os.listdir(classes_home):
        class_root_path = os.path.join(
            classes_home,
            class_name)
        user_to_class_to_imagepath = get_class_images(
            class_root_path,
            user_to_class_to_imagepath)

    print(f"Dumping metadata from {len(user_to_class_to_imagepath)} into data.json")
    with open("data.json", 'w') as f:
        json.dump(user_to_class_to_imagepath, f)
