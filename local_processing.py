import pickle

import numpy as np
import matplotlib.pyplot as plt

from itertools import groupby
from operator import itemgetter
from matplotlib.cbook import flatten
from PIL import Image
from PIL import ImageStat

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
    print(list(zip(val, percentages, count)))
    
    #because there are relatively less recordings of cyclists
    #we calculate how many there are and in which tfRecord they lay
    cyclists = []
    for idx, classes in enumerate(classesval):
        if 4 in classes:
            val, count = np.unique(classes, return_counts=True)
            cyclists.append((count[np.where(val == 4)][0], 
                             data[idx//10]['filename'].numpy().decode("utf-8")))
    first = itemgetter(1)
    sums = [(k, sum(item[0] for item in tups))
        for k, tups in groupby(sorted(cyclists, key=first), key=first)]
    print(sums)


    brightness = []
    for idx, _rec in enumerate(data):
        im = Image.fromarray(extract_from_dump('image')[idx]).convert('L')
        stat = ImageStat.Stat(im)
        brightness.append(stat.rms[0])
        
    
    brightness = np.array(brightness)
    brightness = brightness.reshape(97,10)
    stats=[np.mean(brightness, axis=1),np.std(brightness, axis=1)]

    labels = np.arange(0,len(stats[0]))
    width = 0.8
    fig, ax = plt.subplots()

    ax.bar(labels, stats[0], yerr=stats[1])
    ax.set_ylabel('Mean RMS Brightness')
    ax.set_xlabel('Number for the corresponding tfRecord-file')
    ax.legend()

    plt.show()
    plt.savefig('test5.png', dpi=300)
      
if __name__ == "__main__":
    main()