import pickle
import numpy as np
from itertools import groupby
from operator import itemgetter

from matplotlib.cbook import flatten

def main():
    '''
    In this function some additional EDA is done.
    For easier handling this is done localy with dumped data.
    The dumped data Contains batches of 10 recordings for each tfRecord-file.
    '''
    
    #Loading the dump
    path = 'C:/Users/QXZ0087/Downloads/datadump'
    data = pickle.load(open(path, "rb"))
    
    def extract_from_dump(key):
        values = [a_dict[key].numpy() for a_dict in data]
        return values
    
    #Check if there is a bbox, not within in the Picture
    bboxvalues = extract_from_dump('groundtruth_boxes')
    bboxvalues = list(flatten(bboxvalues))
    #bbox coordinates are expected to be 0.0 => x <= 1.0
    print('Maximum Value for bbox', np.max(bboxvalues))
    print('Minimum Value for bbox', np.min(bboxvalues))

    #Calculate the overall percentage of the different classes
    #This can be taken in to consideration for Split creation,
    #so we have the same Distribution in every set
    classesval = extract_from_dump('groundtruth_classes')
    classesval_flat = list(flatten(classesval))
    val, count = np.unique(classesval_flat, return_counts=True)
    percentages = count/count.sum(axis=0) * 100
    print(list(zip(val,percentages)))
    
    #because there are relatively less recordings of cyclists
    #we calculate how many ther are in which tfRecord
    cyclists = []
    for idx, classes in enumerate(classesval):
        if 4 in classes:
            _val, count = np.unique(classes, return_counts=True)
            cyclists.append((count[len(count)-1], data[idx//10]['filename'].numpy()))

    first = itemgetter(1)
    sums = [(k, sum(item[0] for item in tups_to_sum))
        for k, tups_to_sum in groupby(sorted(cyclists, key=first), key=first)]
         
if __name__ == "__main__":
    main()
