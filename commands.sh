#Install chromium-browser
sudo apt-get update
sudo apt-get install chromium-browser
chromium-browser --no-sandbox

#download pre-trained model and move it in right directory
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz && mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz /home/workspace/experiments/pretrained_model/

#install trash-cli to get rid of trashed items
sudo apt install trash-cli
trash-empty
