import json
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)

def add_class_images(class_root_path, user_to_class_to_imagepath):
    """
    class_root_path is root of class directory
    user_to_class_to_imagepath: defaultdict(lambda: defaultdict(list))
                                is a dictionary mapping user->class->imagepath

    use .mit file's mapping information to add every example of this class to the correct user
    """

    class_hex = os.path.basename(class_root_path)
    class_label = relabel_class(class_hex)
    print(f"Reading class hex {class_hex}, char {chr(int(class_hex, 16))}")
            
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

                # add data to class/user dict
                user_to_class_to_imagepath[user_id][class_label].append(full_image_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--femnist-root', type=str, required=True)
    args = parser.parse_args()

    user_to_class_to_imagepath = defaultdict(lambda: defaultdict(list))
    classes_home = os.path.join(args.femnist_root, "data/raw_data/by_class")
    for class_name in sorted(os.listdir(classes_home)):
        class_root_path = os.path.join(
            classes_home,
            class_name)
        add_class_images(
            class_root_path,
            user_to_class_to_imagepath)

    print(f"Dumping metadata from {len(user_to_class_to_imagepath)} into data.json")
    with open("data.json", 'w') as f:
        json.dump(user_to_class_to_imagepath, f)
