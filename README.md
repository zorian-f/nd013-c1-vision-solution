# Object Detection in an Urban Environment
In this First Project the Task is to train and evaluate a pretrained model from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The [Waymo Open dataset](https://waymo.com/open/) is used as a datasource. To run this Code its highly recommended to ether use the [Dockerfile](build/Dockerfile) or the Udacity Workspace, setting up the Environment locally turns out to be very difficult because of many dependencies. I also did some local Calculating with dumped data, which is explained further down.
The Original starter code can be found at [udacity/nd013-c1-vision-starter](https://github.com/udacity/nd013-c1-vision-starter) and the original Problem instructions can be found in the [README_ORIG.md](README_ORIG.md). The model is trained to detect vehicles, pedestrians and cyclists in an urban environment.

## Exploratory Data Analysis (EDA)
In the EDA we get a better understanding of the Data we are dealing with. The Code lays in the [Exploratory Data Analysis.ipynb](https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/Exploratory%20Data%20Analysis.ipynb).

### Display 10 random Images
The first thing implemented is a simple function `display_instances` which shows 10 random picked pictures off of one tfRecord file. An important note is, that in the tfRecordfiles the BBox Coordinates are normalized and have to be multiplied with height and width.

```python
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
%matplotlib inline

def display_instances(batch):
    """
    This function takes a batch from the dataset and display the image with 
    the associated bounding boxes.
    """

    colormap = {1: 'r', 2: 'g', 4: 'b'}
    f, ax = plt.subplots(10, 1, figsize=(100,100))
    
    for i, rec in enumerate(batch):
    
        x = i % 10
        
        img = rec['image'].numpy()
        bboxes = rec['groundtruth_boxes'].numpy()
        classes = rec['groundtruth_classes'].numpy()
        
        #BBox cooridnates are normalized,
        #Multiplying by image width and heigh
        bboxes[:,(0,2)] *= img.shape[0]
        bboxes[:,(1,3)] *= img.shape[1]
        
        ax[x].imshow(img)
        
        #looping through boxes and displaying them
        for j, box in enumerate(bboxes):
            y1, x1, y2, x2 = box
            rec = Rectangle((x1, y1),
                            (x2- x1),
                            (y2-y1),
                            facecolor='none', 
                            edgecolor=colormap[classes[j]])
            ax[x].add_patch(rec)

        ax[x].axis('off')
    plt.tight_layout()
    plt.show()
```
To get an overall impression also across the different tfRecordfiles we take a random tfRecordfile and also take 10 random recordings within that tfRecord. We accomplish that by shuffling both the paths and the dataset. For better readability only one Picture is shown here, what the exact output looks-like can be seen in [10_samples.png](https://github.com/zorian-f/nd013-c1-vision-solution/blob/43f6f073110518c866b4cbcb5bae566810c64ca9/visualization/10_samples.png).
```python
import random

#shuffling the paths so each time we get images from a different tfRecord
random.shuffle(paths)
dataset = get_dataset(paths[0])
#shuffling the recordings within a tfRecordfile
dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
#take 10 Recordings and display them
batch = dataset.take(10)
display_instances(batch)
```
<p align="center" width="80%">
    <img width="80%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/60b8c2b007e64a51776ae15fb97663264bfde0bf/visualization/1_sample.png"> 
</p>

The Following conclusions can be drawn from a first look at the Pictures:

* The pictures are resized to 640x640 and distorted, they do not have the original aspect Ratio. As we see further down, this resizing leads to a very big number of small Boundingboxes. I would therefore recommend to check wether the pretrained Model can deal with small objects or not. 
* Overall Picturequality is good, sharp and good Lighting. 

### Classdistribution analysis
As an Additional EDA task i analysed the Class distribution along all tfRecordfiles. I took 10 samples from each file and counted the occurrences of the different classes. Besides calculating the 
```python
import pickle

import matplotlib.pyplot as plt
'''
In this Section two things are happening:
-10 recordings are taken from each tfRecord file and the amount per class
 per tfRecord file is calculated to get a sense for the classdistribution 
-We dump de data -> 10*97 = 970 Images, Classes and Boxes to be able to
 process the Data locally
'''

#counts[(vehicle, pedestrian, cyclist)]
counts = np.zeros((len(paths), 3))
#Paths are alphabetically sorted to keep track of which counts belong to which tfRecord
paths.sort()
data = []

for i, path in enumerate(paths):

    print('Proccesing ',path)
    dataset = get_dataset(path)
    dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    batch = dataset.take(10)
    
    for j, rec in enumerate(batch):
        #append recording to dump
        data.append(rec)
        
        classes = rec['groundtruth_classes'].numpy()
        unique, count = np.unique(classes, return_counts=True)
                
        if 1 in unique:
            counts[i, 0] += count[np.where(unique==1)[0][0]]
        if 2 in unique:
            counts[i, 1] += count[np.where(unique==2)[0][0]]
        if 4 in unique:
            counts[i, 2] += count[np.where(unique==4)[0][0]]

#Save the dump to a file
with open("datadump", "wb") as file:
    pickle.dump(data, file)   

#Create Plot to visualize classdistribution
labels = np.arange(0,len(counts))
bar_width = 0.8
fig, ax = plt.subplots()

ax.bar(labels, counts[:, 0], bar_width, label='Vehicle')
ax.bar(labels, counts[:, 1], bar_width, label='Pedestrian')
ax.bar(labels, counts[:, 2], bar_width, label='Cyclist')
ax.set_ylabel('Number of Occurences')
ax.set_xlabel('Number for the corresponding tfRecord-file')
ax.set_title('Occurences of Classes')
ax.legend()

plt.savefig('classdistribution.png')
plt.show()

```
The Resulting Bar Chart shows the stacked occurrences of each class within one tfRecordfile. The Chart shows that the Dataset is very imbalanced, that means that the amount of occurrences of a particular class within one file varies. Simply spoken, there are way more Vehicles than there are Pedestrians and almost no cyclists. This leads to two conclusions:
* A simple mAP validation will give no accurate prediction about the Performance, we should calculate a class-level metric
* In terms of Cross-Validation we have to make sure that the overall occurrence ratios between the classes is maintained within the individual splits.
<p align="center" width="80%">
    <img width="80%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/1a0df4bbab02604a576920e1a94b6245d56d554e/visualization/class_distribution.png"> 
</p>

