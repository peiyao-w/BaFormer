import os
import os.path
import pathlib
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset


import yacs.config
from tqdm import tqdm, trange


def make_dataset(
    directory: str,
    config: yacs.config.CfgNode,
    state: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)

    if state=='train':
        split_file = 'train.split' + str(config.split) + ".bundle"
        file = os.path.join(directory, config.name, "splits", split_file)
    elif state=='val':
        split_file = 'test.split' + str(config.split) + ".bundle"
        file = os.path.join(directory, config.name, "splits", split_file)

    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        gtfname = line.split('\n')[0]
        fname = gtfname.split('.')[0] + '.npy'
        startname = gtfname.split('.')[0] + '.txt'
        fpath = os.path.join(directory, config.name, 'features', fname)
        gtpath = os.path.join(directory, config.name, 'groundTruth', gtfname)
        # startpath = os.path.join(directory, config.name, 'actStart', startname)
        # item = fpath, gtpath, startpath, str(gtfname.split('.')[0])
        item = fpath, gtpath, str(gtfname.split('.')[0])
        instances.append(item)
    f.close()
    return instances


class DatasetFolder(VisionDataset):
    """A data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            config: yacs.config.CfgNode,
            state: str,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        # self.with_noise = config.with_noise
        self.roll = config.roll
        action_to_idx = self._find_classes(os.path.join(self.root, config.name))
        samples = make_dataset(self.root, config, state, action_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.action_to_idx = action_to_idx
        self.samples = samples
        # if state == 'train':
        #     self.samples = samples[ :1]
        # elif state == 'val':
        #     self.samples = samples[ :1]
        self.sample_rate = config.sample_rate
        self.state = state

    def _find_classes(self, dir: str) ->  Dict[str, int]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        f = open(os.path.join(dir, 'mapping.txt'), 'r')
        lines = f.readlines()
        action_to_idx = {}
        for line in lines:
            action_to_idx[line.split()[1]] = int(line.split()[0])
        f.close()
        return action_to_idx

    def loader(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [line.split('\n')[0] for line in lines]
        len_frame = len(content)
        act_sequence = np.zeros(len_frame).astype(np.int32)
        for i in range(len_frame):
            act_sequence[i] = self.action_to_idx[content[i]]
        return act_sequence

    def load_start(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [int(line.split('\n')[0]) for line in lines]
        return content

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.state=='train':
            # act_path, cls_path, start_path, fname = self.samples[index]
            act_path, cls_path,  fname = self.samples[index]

            sample = np.load(act_path)[:, ::self.sample_rate]
            target = self.loader(cls_path)[::self.sample_rate]
            # start = self.load_start(start_path)


            ## generate noise
            noise = np.random.randn(*sample.shape)# add noise herer
            # if self.with_noise is False:
            #     noise = 0

            ## roll for the boundary
            if self.roll is not None:
                roll_dist = int(torch.randint(low=-self.roll, high=self.roll+1, size=(1,)))
                target_roll = np.roll(target, shift=roll_dist, axis=0)
                if roll_dist > 0:
                    target = np.pad(target_roll[roll_dist:], (roll_dist, 0), 'edge')
                elif roll_dist <0:
                    target = np.pad(target_roll[:roll_dist], (0, -roll_dist), 'edge')
            #     a = 1
                ## roll over the act, so need known the boundary, pre-processing
                # change target and sample after roll

            return sample, target, fname, noise


        elif self.state=='val':
            # act_path, cls_path, start_path, fname = self.samples[index]
            act_path, cls_path, fname = self.samples[index]

            sample = np.load(act_path)[:, ::self.sample_rate]
            target = self.loader(cls_path)

            # roll_dist = torch.randint(low=-5, high=6, size=(1,))
            # sample_roll = np.roll(sample, shift=int(roll_dist), axis=1) # 1-->, -1 <--
            # if roll_dist > 0:
            #     sample = np.pad(sample_roll[:, roll_dist: ], ((0, 0), (roll_dist, 0)), 'edge')
            # elif roll_dist <0:
            #     sample = np.pad(sample_roll[:, :roll_dist], ((0, 0), (0, -roll_dist)), 'edge')

        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

            return sample, target, fname

    def __len__(self) -> int:
        return len(self.samples)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'gtea', '50salads', 'breakfast'
    ]:
        if is_train:
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            # train_transform = create_transform(config, is_train=True)
            # val_transform = create_transform(config, is_train=False)
            train_transform = None
            val_transform = None
            train_dataset = DatasetFolder(dataset_dir,config.dataset, 'train', transform=train_transform)
            val_dataset = DatasetFolder(dataset_dir,config.dataset, 'val', transform=val_transform)
            return train_dataset, val_dataset
        else:
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            # val_transform = create_transform(config, is_train=False)
            val_transform = None
            val_dataset = DatasetFolder(dataset_dir,config.dataset, 'val', transform=val_transform)
            return val_dataset

    else:
        raise ValueError()

# ============================================= all ===========================================
class DatasetFolderAll(VisionDataset):
    """A data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            config: yacs.config.CfgNode,
            state: str,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolderAll, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        action_to_idx = self._find_classes(os.path.join(self.root, config.name))
        samples = make_dataset(self.root, config, state, action_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.action_to_idx = action_to_idx
        self.sample_rate = config.sample_rate
        self.state = state

        self.features = []
        self.gt = []
        self.fname = []

        # for i in tqdm(range(100)):
        for i in tqdm(range(len(samples))):
            act_path = samples[i][0]
            cls_path = samples[i][1]
            fname = samples[i][2]

            sample = np.load(act_path)[:, ::config.sample_rate]
            if state == "train":
                target = self.loader(cls_path)[::config.sample_rate]
            elif state == "val":
                target = self.loader(cls_path)
            self.features.append(sample)
            self.gt.append(target)
            self.fname.append(fname)

    def _find_classes(self, dir: str) ->  Dict[str, int]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        f = open(os.path.join(dir, 'mapping.txt'), 'r')
        lines = f.readlines()
        action_to_idx = {}
        for line in lines:
            action_to_idx[line.split()[1]] = int(line.split()[0])
        f.close()
        return action_to_idx

    def loader(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [line.split('\n')[0] for line in lines]
        len_frame = len(content)
        act_sequence = np.zeros(len_frame).astype(np.int32)
        for i in range(len_frame):
            act_sequence[i] = self.action_to_idx[content[i]]
        return act_sequence


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample= self.features[index]
        target= self.gt[index]
        fname = self.fname[index]
        return sample, target, fname

    def __len__(self) -> int:
        return len(self.features)

def create_dataset_all(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'gtea', '50salads', 'breakfast'
    ]:
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = None
        val_transform = None
        train_dataset = DatasetFolderAll(dataset_dir,config.dataset, 'train', transform=train_transform)
        val_dataset = DatasetFolderAll(dataset_dir,config.dataset, 'val', transform=train_transform)
        return train_dataset, val_dataset
    else:
        raise ValueError()
