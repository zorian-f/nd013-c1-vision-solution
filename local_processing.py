import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from matplotlib.cbook import flatten
from matplotlib.patches import Rectangle
from PIL import Image
from PIL import ImageStat

def main():
    '''
    In this function some additional EDA is done.
    For easier handling this is done localy with dumped data.
    The dumped data Contains batches of 10 recordings for each tfRecord-file.
    '''
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
    return

def plot_mean_rms_brightness():
    '''
    This function calculates the mean RMS brightness for every set of 10 pictures
    creates a plo twhith specific STD values 
    '''
    brightness = []
    images = extract_from_dump('image')
    for idx, _rec in enumerate(data):
        im = Image.fromarray(images[idx]).convert('L')
        stat = ImageStat.Stat(im)
        brightness.append(stat.rms[0])
    brightness = np.array(brightness)
    brightness = brightness.reshape(97,10)
    stats=[np.mean(brightness, axis=1),np.std(brightness, axis=1)]

    #Creating plot
    labels = np.arange(0,len(stats[0]))
    width = 0.8
    fig, ax = plt.subplots()

    ax.bar(labels, stats[0], alpha=0.5, color='blue', edgecolor='black', yerr=stats[1])
    ax.set_ylabel('Mean RMS Brightness')
    ax.set_xlabel('Number for the corresponding tfRecord-file')
    ax.legend()

    plt.show()
    plt.savefig('mean_rms_brightness.png', dpi=300)
    return


def display_instance(number):
    '''
    This Function takes a Number and Plots/saves the realated anotated picture
    '''
    colormap = {1: 'r', 2: 'g', 4: 'b'}
    my_dpi=96
    fig, ax = plt.subplots(figsize=(832/my_dpi, 832/my_dpi), dpi=my_dpi)
    
    img = extract_from_dump('image')[number]
    bboxes = extract_from_dump('groundtruth_boxes')[number]
    classes = extract_from_dump('groundtruth_classes')[number]
    filename = extract_from_dump('filename')[number].decode("utf-8")
    
    #BBox cooridnates are normalized,
    #Multiplying by image width and heigh
    bboxes[:,(0,2)] *= img.shape[0]
    bboxes[:,(1,3)] *= img.shape[1]
    
    ax.imshow(img)
    ax.axis('off')
        #looping through boxes and displaying them
    for j, box in enumerate(bboxes):
        y1, x1, y2, x2 = box
        rec = Rectangle((x1, y1),
                        (x2- x1),
                        (y2-y1),
                        facecolor='none', 
                        edgecolor=colormap[classes[j]])
        ax.add_patch(rec)
           
    plt.savefig(np.str(number)+'_'+filename+'.png', bbox_inches='tight', pad_inches = 0, dpi=my_dpi)
  
def plot_bbox_histogramm():
    '''
    This Function Plots a Histogram which shows the Distrubtion ob BBox-size [square-Pixels]
    '''
    records = extract_from_dump('groundtruth_boxes')
    height = []
    width = []
    area = []
    for idx, bboxes in enumerate(records):
        bboxes *= 640
        for box in bboxes:
            y1, x1, y2, x2 = box
            height.append(y2-y1)
            width.append(x2-x1)
            area.append((y2-y1)*(x2-x1))
    
    data = area
    my_dpi=300
    bin = 100
    bar_color = 'cyan'
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for idx, a in enumerate(ax):
        a.set_yscale('log')
        a.grid(which="both",alpha=0.4)
        a.set_axisbelow(True)
    
    ax[0].hist(data, bins=bin, alpha=0.5, facecolor=bar_color, histtype='bar', ec='black')
    ax[1].hist(data, bins=bin, alpha=0.5, facecolor=bar_color, histtype='bar', ec='black', range=[0, 10000])
    ax[2].hist(data, bins=bin, alpha=0.5, facecolor=bar_color, histtype='bar', ec='black', range=[0, 1000])

    fig.text(0.5, 0.01, 'Pixel-Area [pixel$^2$]', ha='center')
    fig.text(0.09, 0.5, 'Number of occurrences', va='center', rotation='vertical')
    
    plt.savefig('bbox_size_histo.png', bbox_inches='tight', pad_inches = 0.1, dpi=my_dpi)
    return
    
  
def extract_from_dump(key):
    '''
    Helper Function which extract Data from the Dump
    '''
    values = [a_dict[key].numpy() for a_dict in data]
    return values
    
if __name__ == "__main__":

    #Loading the dump
    path = 'PATH TO DUMPFILE'
    data = pickle.load(open(path, "rb"))
    
    #main()
    #plot_mean_rms_brightness()
    #display_instance(649)
    plot_bbox_histogramm()
