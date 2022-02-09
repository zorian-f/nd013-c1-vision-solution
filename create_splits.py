import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.

    train_dir = data_dir+'/train'
    val_dir = data_dir+'/val'
    train_and_val_dir = data_dir+'/training_and_validation'
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    files = glob.glob(train_and_val_dir+'/*')
    
    #For the first try we split the data naively in 80%-20% train and validation
    random.shuffle(files)
    train_size = int(len(files)*0.8)
    train_files = files[:train_size]
    eval_files = files[train_size:]

    #delete all exisitng files/splits
    del_files = glob.glob(train_dir+'/*') + glob.glob(val_dir+'/*')
    if del_files:
        for file in del_files:
            os.remove(file)


    #Create symlinks for the splitdata
    for files in train_files:
        os.symlink(files, train_dir+'/'+os.path.basename(files))
    
    for files in eval_files:
        os.symlink(files, val_dir+'/'+os.path.basename(files))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', default='/home/workspace/data/waymo',
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
