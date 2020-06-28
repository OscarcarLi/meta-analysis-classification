from torchvision import transforms
import glob
import pickle
import random
from collections import defaultdict
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from maml.models.gated_conv_net_original import ImpRegConvModel



class MiniimagenetMAMLSplit():
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, label_correction=0, **kwargs):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root + '/miniimagenet'

        self._split = split
        self._label_correction = label_correction

        if self._split == 'test':
            print('MiniImagenet test')
            all_character_dirs = glob.glob(self.root + '/test/**')
            self._characters = all_character_dirs
        elif self._split == 'val':
            print('MiniImagenet val')
            all_character_dirs = glob.glob(self.root + '/val/**')
            self._characters = all_character_dirs
        else:
            print('MiniImagenet train')
            all_character_dirs = glob.glob(self.root + '/train/**')
            self._characters = all_character_dirs

        self._character_images = []
        for i, char_path in enumerate(self._characters):
            img_list = [(cp, i) for cp in glob.glob(char_path + '/*')]
            self._character_images.append(img_list)

        self._flat_character_images = sum(self._character_images, [])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target
            character class.
        """
        image_path, character_class = self._flat_character_images[index]
        image = Image.open(image_path, mode='r')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        character_class += self._label_correction

        return image, character_class

    def __len__(self):
        return len(self._flat_character_images)


def dump_features_for_dataset(model, dataset):
    
    loader = DataLoader(dataset, 
        batch_size=100, num_workers=4, pin_memory=True, 
        shuffle=False, sampler=None)

    features_dict = defaultdict(list)

    for batch in tqdm(loader, total=len(loader)):
        X, y = batch
        X = X.cuda()
        feat = model(X, modulation=None)
        for i, label in enumerate(y.numpy()):
            features_dict[label].append(feat[i].unsqueeze(0).detach().cpu())
        
    for label in features_dict.keys():
        features_dict[label] = torch.cat(features_dict[label], dim=0).numpy()
        # print(f"Got features of shape {features_dict[label].shape} for label {label}")

    return features_dict



if __name__ == '__main__':
    
    # chkpt and model load
    model = ImpRegConvModel(
        input_channels=3,
        num_channels=64,
        img_side_len=84,
        verbose=True,
        retain_activation=True,
        use_group_norm=True,
        add_bias=False)
    chkpt = sys.argv[1]
    print(f"loading chkpt from {chkpt}")
    state_dict = torch.load(chkpt)
    model.load_state_dict(state_dict['model'])
    model.to('cuda')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # dataset creation
    resize = transforms.Resize(
        (84, 84), Image.LANCZOS)
    img_transform = transforms.Compose(
        [resize, transforms.ToTensor()])
    datasets = {
        'train': MiniimagenetMAMLSplit(
            'data',  split='train', transform=img_transform, download=True, label_correction=0),
        'val': MiniimagenetMAMLSplit(
            'data',  split='val', transform=img_transform, download=True, label_correction=64),
        'test': MiniimagenetMAMLSplit(
            'data',  split='test', transform=img_transform, download=True, label_correction=80)
    }


    # forward
    features = {}
    for split, dataset in datasets.items():
        print(f"processing split {split}")
        features[split] = dump_features_for_dataset(model, dataset)

    
    # dump features
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(features, f)