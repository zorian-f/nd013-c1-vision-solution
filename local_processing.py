import pickle
import numpy as np

from matplotlib.cbook import flatten

def main():
    '''
    In this function some additional EDA is done.
    For easier handling this is done localy with dumped data.
    The dumped data Contains batches of 10 recordings for each tfRecord-file.
    '''
    
    #Loading the dump
    path = 'PATH TO DUMP'
    data = pickle.load(open(path, "rb"))

    #Check if there is a bbox, not within in the Picture
    a_key = 'groundtruth_boxes'
    values_of_key = [a_dict[a_key].numpy() for a_dict in data]
    values_of_key = list(flatten(values_of_key))
    #bbox coordinates are expected to be 0.0 => x <= 1.0
    print('Maximum Value for bbox', np.max(values_of_key))
    print('Minimum Value for bbox', np.min(values_of_key))

    
if __name__ == "__main__":
    main()
