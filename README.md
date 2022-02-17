# Object Detection in an Urban Environment
In this First Project the Task is to train and evaluate a pretrained model from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The [Waymo Open dataset](https://waymo.com/open/) is used as a datasource. To run this Code its highly recommended to ether use the [Dockerfile](build/Dockerfile) or the Udacity Workspace, setting up the Environment locally turns out to be very difficult because of many dependencies. I also did some local Calculating with dumped data, which is explained further down.
The Original starter code can be found at [udacity/nd013-c1-vision-starter](https://github.com/udacity/nd013-c1-vision-starter) and the original Problem instructions can be found in the [README_ORIG.md](README_ORIG.md). The model is trained to detect vehicles, pedestrians and cyclists in an urban environment.

## Exploratory Data Analysis (EDA)
In the EDA we get a better understanding of the Data we are dealing with. The Code lays in the [Exploratory Data Analysis.ipynb](https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/Exploratory%20Data%20Analysis.ipynb). The pre-installed Firefox browser is crashing all the time, therefore its suggested to install and use chromium-browser with following shell-commands:
```shell
sudo apt-get update
sudo apt-get install chromium-browser
chromium-browser --no-sandbox
```

### Display 10 random Images
The first thing implemented is a simple function `display_instances` which shows 10 random picked pictures off from one tfRecord file. An important note is, that in the tfRecordfiles the BBox Coordinates are normalized and have to be multiplied with height and width.

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

* The pictures are resized to 640x640 and distorted, they do not have the original aspect Ratio. As we see further down, this resizing leads to a very big number of small Bounding boxes. I would therefore recommend to check whether the pretrained Model can deal with small objects or not. 
* Overall Image quality is good, sharp and good Lighting. 

### Class Distribution analysis
As an Additional EDA task I analyzed the Class Distribution along all tfRecordfiles. I took 10 samples from each file and counted the occurrences of the different classes. Besides calculating the Class Distribution I dump all the recordings into a dump-file which I later use to for local processing.
```python
import pickle

import matplotlib.pyplot as plt
'''
In this Section two things are happening:
-10 recordings are taken from each tfRecord file and the number of class occurrences
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
The Resulting Bar Chart shows the stacked occurrences of each class within one tfRecordfile. The Chart shows that the Dataset is very imbalanced, that means that the amount of occurrences of a particular class within one file varies. Simply spoken, there are way more Vehicles than there are Pedestrians and almost no cyclists. This leads the following conclusions:
* A simple mAP validation will give no accurate prediction about the Performance, we should calculate a class-level metric.
* In terms of Cross-Validation we must make sure that the overall occurrence ratio between the classes is maintained within the individual splits. We also must make sure that all classes even occur in our Splits, with the sparse occurrence of cyclists this could be difficult if the splits are randomly picked. I suggest using a stratified cross validation like a stratified KFold.
* To Improve Training and avoid bias towards one class, one could use oversampling.

<p align="center" width="80%">
    <img width="80%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/1a0df4bbab02604a576920e1a94b6245d56d554e/visualization/class_distribution.png"> 
</p>

### Local Processing
All the analyses shown in this Section were done locally by using the dumped raw data extracted from the tfRecord files as shown in the Previous section. All the local processing is done in the [local_processing.py](local_processing.py).
#### Boundingbox check
In `check_bbox()` the Coordinates of every Boundingbox is checked on whether they are within a range of 0.0 - 1.0. There is no problem with the BBox Coordinates as can be seen by the output:
```
Maximum Value for bbox 1.0
Minimum Value for bbox 0.0
```
#### Analyze Cyclists
As shown in the Class Distribution analysis, there is an imbalance in classes. Especially the cyclsit class is very underrepresented in the dataset. To know a exact percentage of the proportion of each class I calculated the overall percentage per Class in `analyse_cyclists()`, I also analysed which tfRecord file holds the cyclist class und how much of them.
```
[(1, 76.3081267096091, 17296), (2, 23.07861995941057, 5231), (4, 0.613253330980323, 139)]
('segment-10023947602400723454_1120_000_1140_000_with_camera_labels_10.tfrecord', 2)
('segment-10023947602400723454_1120_000_1140_000_with_camera_labels_70.tfrecord', 1)
('segment-1005081002024129653_5313_150_5333_150_with_camera_labels_100.tfrecord', 2)
('segment-1005081002024129653_5313_150_5333_150_with_camera_labels_140.tfrecord', 5)
('segment-1005081002024129653_5313_150_5333_150_with_camera_labels_180.tfrecord', 7)
('segment-10061305430875486848_1080_000_1100_000_with_camera_labels_140.tfrecord', 4)
('segment-10061305430875486848_1080_000_1100_000_with_camera_labels_20.tfrecord', 4)
('segment-10061305430875486848_1080_000_1100_000_with_camera_labels_30.tfrecord', 13)
('segment-10061305430875486848_1080_000_1100_000_with_camera_labels_50.tfrecord', 4)
('segment-10072140764565668044_4060_000_4080_000_with_camera_labels_180.tfrecord', 9)
('segment-10072140764565668044_4060_000_4080_000_with_camera_labels_30.tfrecord', 2)
('segment-10075870402459732738_1060_000_1080_000_with_camera_labels_70.tfrecord', 15)
('segment-10082223140073588526_6140_000_6160_000_with_camera_labels_0.tfrecord', 23)
('segment-10082223140073588526_6140_000_6160_000_with_camera_labels_170.tfrecord', 2)
('segment-10082223140073588526_6140_000_6160_000_with_camera_labels_180.tfrecord', 2)
('segment-10082223140073588526_6140_000_6160_000_with_camera_labels_50.tfrecord', 5)
('segment-10094743350625019937_3420_000_3440_000_with_camera_labels_100.tfrecord', 5)
('segment-10094743350625019937_3420_000_3440_000_with_camera_labels_120.tfrecord', 5)
('segment-10094743350625019937_3420_000_3440_000_with_camera_labels_130.tfrecord', 5)
('segment-10096619443888687526_2820_000_2840_000_with_camera_labels_30.tfrecord', 6)
('segment-10096619443888687526_2820_000_2840_000_with_camera_labels_90.tfrecord', 3)
('segment-10107710434105775874_760_000_780_000_with_camera_labels_190.tfrecord', 10)
('segment-10107710434105775874_760_000_780_000_with_camera_labels_70.tfrecord', 5)
```
As can bee seen by the first line of the output `[(1, 76.3081267096091, 17296), (2, 23.07861995941057, 5231), (4, 0.613253330980323, 139)]` there are only 0.61% of cyclists, 23.08% pedestrians and 76.30% Vehicles. The result of the analysis confirms the imbalance, which we already saw in the Class distribution analysis. The rest of the output shows how many recordings of a cyclists are within one tfRecord file, the last line for exmaple shows that there are 5 occurences of cyclist int `segment-10107710434105775874_760_000_780_000_with_camera_labels_70.tfrecord`.
#### Boundingbox Size
The Images of the Dataset were heavily resized and distorted. Along the pictures, the Boundingboxes got resized as well and therefore Boxes which were small in the first place got even smaller. The get a good Impression of the Size distribution I created a Histogram which shows the distribution of Box sizes (squarepixels). The three Histograms are from the same data, only plotted with different ranges.

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/bbox_size_histo.png"> 
</p>

The left graph shows the data with maximum range, what stands out is that there is a notably big Boundginbox (>400k) If we sample a pictures from that Dataset we can see that there is a error in the dataset. The Recording suggests that there is a pedestrian over the whole screen which makes no sense. Because of that error the `segment-11252086830380107152_1540_000_1560_000_with_camera_labels.tfrecord` should no be used. 

<p float="left" align="middle" width="49%" >
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/642_segment-11252086830380107152_1540_000_1560_000_with_camera_labels_50.tfrecord.png" width="49%"/>
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/646_segment-11252086830380107152_1540_000_1560_000_with_camera_labels_180.tfrecord.png" width="49%"/> 
</p>

The Histogram also shows that there are a lot of "small" BBoxes (note the logarithmic scale!). In the Pictures below we can see examples of small BBoxes. As stated in the the [SSD Paper](https://arxiv.org/pdf/1512.02325.pdf), the model got a bad performance at small objects and is very sensitive to BBox-size:
>... Figure 4 shows that SSD is very sensitive to the bounding box size. In other words, it has much worse performance on smaller objects than bigger objects. ...

The Paper also suggests random cropping as a augmentation method to increase the performance on small objects:
> ...  The random crops
generated by the strategy can be thought of as a ”zoom in” operation and can generate
many larger training examples. To implement a ”zoom out” operation that creates more
small training examples, we first randomly place an image on a canvas of 16× of the
original image size filled with mean values before we do any random crop operation. ... We have seen a consistent
increase of 2%-3% mAP across multiple datasets, as shown in Table 6. In specific, Figure 6 shows that the new augmentation trick significantly improves the performance on
small objects. This result underscores the importance of the data augmentation strategy
for the final model accuracy. ...
 
In the left Picture two reference-boxes can be seen, to get a better understanding of the sizes.

<p float="left" align="middle" width="49%" >
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/109_segment-10153695247769592104_787_000_807_000_with_camera_labels_30.tfrecord.png" width="49%"/>
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/312_segment-10596949720463106554_1933_530_1953_530_with_camera_labels_0.tfrecord.png" width="49%"/> 
</p>


#### Image Brightness
To get an overall impression of image Brightness, I calculated the mean RMS brightness for every tfRecordfile with `plot_mean_rms_brightness()`.
<p align="center" width="80%">
    <img width="80%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/d4cd08871ebceb09028b0afe695124505d2ac5a5/visualization/mean_rms_brightness.png"> 
</p>

What stands out is that there are some dips in brightness, for examaple in Dataset-number 6 and 37. When we take a sample Picture with `display_instance()` from those Datasets we can see that those records were taken at nighttime. Even though there are some recordings made at nighttime, most of the Dataset is taken in broad daylight. Because of the underrepresentation of nighttime recordings, maybe an augmentation that turns down brightness could improve nighttime performance.

<p float="left" align="middle" width="49%" >
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/371_segment-10724020115992582208_7660_400_7680_400_with_camera_labels_130.tfrecord.png" width="49%"/>
  <img src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/main/visualization/61_segment-10082223140073588526_6140_000_6160_000_with_camera_labels_0.tfrecord.png" width="49%"/> 
</p>

## Training
In this Section we train and evaluate the model, first we must download the pretrained model and move it in the right direction, this is done by the following shell-command:
```shell
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz && mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz /home/workspace/experiments/pretrained_model/
```
I also noticed that when I delete files, e. g. training data over the "desktop" interface, files dont get removed completely, they are moved to trash. I found it helpful to use `trash-cli` to get rid of them:
```shell
sudo apt install trash-cli
trash-empty
```
All the shell commands can be found in [commands.sh](commands.sh). All the pipeline configs can be found in [/experiments/pipeline_configs/](https://github.com/zorian-f/nd013-c1-vision-solution/tree/main/experiments/pipeline_configs).

### Create Splits
First we have to create splits, I did choose a ratio of 80% Training and 20% validation data (this is a common ratio as learned in class). I do not create a test-split because these are already given. I do create symbolic links of the data to save space. The code for creating splits can be found in [create_splits.py](create_splits.py):
```python
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
    
    # check if folders exist and create them if not
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    files = glob.glob(train_and_val_dir+'/*')
    
    # we split the data naively in 80%-20% train and validation
    random.shuffle(files)
    train_size = int(len(files)*0.8)
    train_files = files[:train_size]
    eval_files = files[train_size:]

    # delete all exisitng files/splits
    del_files = glob.glob(train_dir+'/*') + glob.glob(val_dir+'/*')
    if del_files:
        for file in del_files:
            os.remove(file)


    # Create symlinks for the splitdata
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
```
### Refference Run
The first run gives a refference to compare to. I generate a `pipeline_new.config` by executing the `edit_config.py` with the correspoding directorys:
```shell
python edit_config.py --train_dir /home/workspace/data/waymo/train/ --eval_dir /home/workspace/data/waymo/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
I start the training and let it run for a short time:
```shell
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
What stands out imidiately is that the loss is bouncing and very high:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/629a8ea0747752ff78d61bfba3afc40a9ec8be27/visualization/traing_and_val/reff_run_1.png"> 
</p>

As suggested in class, a bouncing loss-function means that the learning rate is too high, so I did stop the training and also didn't evaluate. Instead I tried adjusting the learning-rate, I did so by changing the corresponding values in the `pipeline_new.config`:
```
learning_rate_base: 0.004           # previuos 0.04
warmup_learning_rate: 0.0013333     # previuos 0.0133
```
<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/2f27f3f71568962f4d44134f9645997be5280634/visualization/traing_and_val/reff_run_2_adjusted_lr.png"> 
</p>

With the adjusted learning rate the total-loss is decreasing much faster, after 1k Steps its already at 0.8 whereas the previous run was at somewhere around 5. With the adjusted Parameters i did a run with evaluation. There is a "bump" in the data, this is due to an interruption of the evaluation (stopping and resuming it), later graphs won't have this bump. 

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/16240ee5a5a024c830485e2b3f74ea81a65502dd/visualization/traing_and_val/reff_run_3_1.png">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/16240ee5a5a024c830485e2b3f74ea81a65502dd/visualization/traing_and_val/reff_run_3_2.png">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/16240ee5a5a024c830485e2b3f74ea81a65502dd/visualization/traing_and_val/reff_run_3_3.png">
</p>

### Experiment 1
While doing the Reference run I noticed that the train-loop is using up almost all GPU memory (check with `nvidia-smi` in terminal), this can lead to not having enough memory for running the evaluation-loop simultaneously. TensorFlow gives us an option to limit how much GPU memory is used ([TensorFlow Guide](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)). As suggested in the Guide i added the following code to the `model_main_tf2_py`:
```Python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 9GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=9000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
```
I found 9GB to be sufficient for the training-loop and leave the rest for evaluation. The code can be found in [model_main_tf2_mod.py](experiments/model_main_tf2_mod.py). In this experiment, the only Parameter I changed is `batch_size: 5`:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/84fbcee0777da993a0db850d347a695245290749/visualization/traing_and_val/exp1_run_1.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/84fbcee0777da993a0db850d347a695245290749/visualization/traing_and_val/exp1_run_2.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/84fbcee0777da993a0db850d347a695245290749/visualization/traing_and_val/exp1_run_3.PNG">
</p>

This yields to a very good result, the training- and evaluation loss are bove converging very fast to a Value of 0.5. Also the distance between the training- and validation loss is small. We can also See that as expected, the model performs better at large objects rather than small ones:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.170
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.148
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.046
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
```
The Picture bellow is taken from the TensorBoard and shows the Prediction on the left and the Ground truth to the right. We can see that the model struggles to detect smaller objects. As mentioned in the EDA Section, this is a weak point with this model.

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/84fbcee0777da993a0db850d347a695245290749/visualization/traing_and_val/side_by_side1.png">
</p>

### Experiment 2
For this experiment I use augmentation to try improve the results, to visualize the used methods I use the [Jupyter Notebook](https://github.com/zorian-f/nd013-c1-vision-solution/blob/3d606565d282bfb3017939463f6745b0db87837e/Explore%20augmentations.ipynb). As the [SSD Paper](https://arxiv.org/pdf/1512.02325.pdf) suggests, random cropping should improve the performance for smaller objects. In the EDA we also saw that there are nighttime recordings, so I did also add a brightness adjusting method. There are several built-in augmentation methods that come with the TensorFlow Object Detection API the, [here](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) is a full list. I did choose to use the following methods:
```Python
random_adjust_brightness
ssd_random_crop
random_vertical_flip
```
The augmentation did not change the Result:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e76fe0c7c96b2bf8c4bffed5ce65869e8b3f12e5/visualization/traing_and_val/exp2_run_1.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e76fe0c7c96b2bf8c4bffed5ce65869e8b3f12e5/visualization/traing_and_val/exp2_run_2.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e76fe0c7c96b2bf8c4bffed5ce65869e8b3f12e5/visualization/traing_and_val/exp2_run_3.PNG">
</p>

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.172
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
```
### Experiment 3
In this Experiment I switched the momentum optimizer with adam. I did so by adding the following lines to my `pipeline_new.config`:
```
optimizer {
    # momentum_optimizer {
    adam_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: .0001
          schedule {
            step: 500
            learning_rate: .0002
          }
          schedule {
            step: 1000
            learning_rate: .00004
          }
          schedule {
            step: 1500
            learning_rate: .00002
          }
        }
      }
      # momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
I tried using a sheduler and after 160 Steps I stopped because there was no change in the loss:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/abad7653f04958f98f23ee1e5105f6cbacd9c806/visualization/traing_and_val/exp3_run_1.PNG">
</p>

I then changed the sheduler to reduce learnin rate early:
```
          schedule {
            step: 500
            learning_rate: .0002
          }
          schedule {
            step: 1000
            learning_rate: .00004
          }
          schedule {
            step: 1500
            learning_rate: .00002
          }
```
I also changed the `model_main_tf2.py` to print out checkpoints more often:
```Python
flags.DEFINE_integer(
    'checkpoint_every_n', 100, 'Integer defining how often we checkpoint.')
```
This gives me the following Result:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/abad7653f04958f98f23ee1e5105f6cbacd9c806/visualization/traing_and_val/exp3_run_2.PNG">
</p>

The delta bewteen training- and validationloss is big, this gives not a good result. The training converges very quickly maybe I am hitting a local minimum very early. I next tried to use the the cosine decay as I did with the momentum optimizer:
```
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 0.002
          total_steps: 600
          warmup_learning_rate: 0.0013333
          warmup_steps: 50
        }
      }
```
This gives better results but still not as good as with the momentum optimizer:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e53051ed886f0d5915d98e7882111e50ec1b165c/visualization/traing_and_val/exp3_run_3_1.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e53051ed886f0d5915d98e7882111e50ec1b165c/visualization/traing_and_val/exp3_run_3_2.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/e53051ed886f0d5915d98e7882111e50ec1b165c/visualization/traing_and_val/exp3_run_3_3.PNG">
</p>

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.040
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.067
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.023
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.352
```
### Experiment 4
For the last experiment I wanted to rerun the second experiment to see if i can get better results, this time i let it run for 3500 steps.

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/ff4d2042bd4473860fa590c85a8f9b69ec55e40f/visualization/traing_and_val/exp4_run_1.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/ff4d2042bd4473860fa590c85a8f9b69ec55e40f/visualization/traing_and_val/exp4_run_2.PNG">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/ff4d2042bd4473860fa590c85a8f9b69ec55e40f/visualization/traing_and_val/exp4_run_3.PNG">
</p>

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.135
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.046
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.229
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.562
```
This run did not improve over experiment 2.

### Inference Video
I used the model from my last experiment to create the inference video:

<p align="center" width="100%">
    <img width="100%" src="https://github.com/zorian-f/nd013-c1-vision-solution/blob/5121bdb4b5ed8e1a78fc7505cd88fe10c054ac7d/visualization/animation.gif">
</p>

