### Project overview
In this First Project the Task is to train and evaluate a pretrained modle from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The [Waymo Open dataset](https://waymo.com/open/) is used as a datasource.
The Original starter code can be found at [udacity/nd013-c1-vision-starter](https://github.com/udacity/nd013-c1-vision-starter) and the original Problem instructions can be found in the [README_ORIG.md](README_ORIG.md). The modle is trained to detect vehicles, pedestrians and cyclists in an urban environment.

### Exploratory Data Analysis (EDA)
In the EDA we get a better understanding of the Data we are dealing with. The Code lays in the [Exploratory Data Analysis.ipynb](https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/Exploratory%20Data%20Analysis.ipynb).
The first thing implented is a simple function `display_instances` which shows 10 random picked pictures off of one tfRecord file. An important note is, that in the tfRecordfiles the BBox Coordinates are noramlized and have to be multiplied with height and width.

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
To get an overall impression also across the different tfRecordfiles we take a random tfRecordfile and also take 10 random recordings within that tfRecord. We accomplish that by shuffling both the paths and the dataset.
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
