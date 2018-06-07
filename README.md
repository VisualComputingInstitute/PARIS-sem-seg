# Semantic Segmentation
A somewhat clean implementation of a straight forward semantic segmentation
network in TensorFlow. This is mainly used for the BMBF project PARIS.

Large parts correspond to the ResNet baseline described in:
[Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323)
Tobias Pohlen, Alexander Hermans, Markus Mathias, and Bastian Leibe. CVPR 2017
The original Theano code can be found [here](https://github.com/TobyPDE/FRRN).

A big chunk of this code has been copied or adapted from our
[person re-identification project](https://github.com/VisualComputingInstitute/triplet-reid).
This code was written by Lucas Beyer and me, however, he did most of the code
polishing.

## Usage
Some quick pointers on how to train and evaluate the network. Most arguments
have a decent documentation. Please refer to those for more details

### Dataset Generation
First downscale the images for realistic training. Be aware this will be
resizing a lot of images. Since resizing of labels was not implemented super
efficiently this will take several hours, but it is less wasteful than doing it
on the fly during training.

Example call:
```
python resize_cityscapes.py --downscale_factor 4 \
        --cityscapes_root /work/hermans/cityscapes \
        --target_root /work/hermans/cityscapes/down4
```

### Training
Make sure the dataset is available in the desired downsample rate. My currently
best working setup for CityScapes is the following:
```
python train.py --experiment_root /root/to/store/experiment_data \
        --train_set datasets/cityscape_fine/train.csv \
        --dataset_root /root/to/downsampled/data \
        --dataset_config datasets/cityscapes.json \
        --flip_augment \
        --gamma_augment \
        --fixed_crop_augment_height 192 \
        --fixed_crop_augment_width 448 \
        --loss_type bootstrapped_cross_entropy_loss \
```
This should give about ~70% mean IoU on the validation set.

### Evaluation
The evaluation script is used for two things. It passes all the images provided
in a csv file through the network and it performs the numerical evaluation. At
the same time it can be used to save color coded or grayscale result images.
The typical CityScapes evaluation (+ storing RGB images) can be done using:
```
python evaluate.py --experiment_root /root/to/store/experiment_data \
        --eval_set datasets/cityscape_fine/val.csv \
        --rgb_input_root /root/to/downsampled/data \
        --full_res_label_root /root/to/raw/cityscapes \
        --save_predictions full
```
It's important to notice that this script also needs the path to the original
CityScapes images. The original image size is used for evaluation and this is
parse from the original images, as well as the ground truth labels.

Additionally, there is a little tool to simply run the network on arbitrary
images:
```
python predict.py --experiment_root /root/to/store/experiment_data \
        --image_glob /some/image/pattern*.jpg \
        --result_directory /path/to/store/images \
        --rescale_w 512
        --rescale_h 256
```

## Software requirements
This code was tested with both TensorFlow 1.4 and 1.8. The latter, using up to date CUDA and cudnn versions runs significantly faster (about 1.4 on a Titan X Pascal). It was only tested with Python 3.5/3.6. Some of the code defenitely doesn't work like expected with Python 2.7. Other packages aren't critical as long as they are up-to-date enough to run with the used TensorFlow version.

## ROS nodes
There is a very thin wrapper around a trained frozen network. Please check the corresponding [README](https://github.com/VisualComputingInstitute/PARIS-sem-seg/blob/master/ros_nodes/ROS_NODES_README.md). This of course runs on Python 2.7, but there are no guarantees for the rest of the code.

### Pretrained frozen models for the ROS nodes
In order to run the semantic segmentation ros node a frozen graph is needed. To do some initial debugging, here are two frozen graphs that could be used. While both models are fully convolutional, the scale at which objects appear in the images should reoughly correspond to the scale the network was trained on.
* [Full model](https://rwth-aachen.sciebo.de/s/1BJk7Ek5XrA5vvI) This is a default model as used in most experiments and it was trained on the Mapillary Dataset, where all images were rescaled to a width of 512 pixels. This model runs at around 1.4 fps on a Jetson TX2 using 256x512 inputs.
* [Reduced model](https://rwth-aachen.sciebo.de/s/UIvpPJtcGOszjOd) This model is trained on quater resolution CityScapes images and uses a reduced base channel count (8 instead of 48), hence its performance is significantly degraded. However, this model runs at almost 11 fps on a Jetson TX2 using 256x512 inputs.

## Multi Dataset Training
Training on multiple datasets is now supported. This means you can simply add a
list of arguments which specify the dataset and train jointly on them.
Every dataset gets a separate logit output and a loss is computed only for the
dataset specific logit. For example:
```
python train.py --experiment_root /root/to/store/experiment_data \
        --train_set datasets/cityscape_fine/train.csv datasets/mapillary/train.csv\
        --dataset_root /root/to/downsampled_cs/data /root/to/downsampled_mpl/data\
        --dataset_config datasets/cityscapes.json datasets/mapillary.json \
        --flip_augment \
        --gamma_augment \
        --fixed_crop_augment_height 192 \
        --fixed_crop_augment_width 448 \
        --loss_type bootstrapped_cross_entropy_loss \
```
In order to make these changes the old crop augmentation is now removed and
replaced by a new one that takes fixed size crops from the image. This was
needed for mapillary training already. Additionally, the old logs now no longer
really work for resuming or evaluation.

## Migrating old experiments
When trying to use newer versions of the code with older experiments (e.g. for
timing or evaluation) this will not always work. For a limited support one thing
to try is to migrate the `args.json` with the `migrate_args.py` script. This
will fix some of the issues, but it is not guaranteed to work. Especially the
checkpoints are not affected by this migration which would need to be the case
for flawless support.

## Design space
A list of things to try eventually to see how the performance and speed changes.

* Convolution type
    * Normal convolutions
    * Depthwise seperable convolutions
    * Bottle neck resblocks
* Depth multiplier
* Resblock count
* Pooling counts
* PSP net style blocks in network center
* Completely different network architectures



## Main Todo
* Replace printing with logging
* Pull together the eval, predict and freeze script and use the freeze script in
eval and predict so that code is less redundant and the model is frozen automatically.
* Upload timing script.
* Add an input loader and augmentation visualization script.
* Add parametrization to the network (which convs, which blocks etc. as seen above.)
* Add an E-net, or a mobile net, this might also include loading pretrained weights.
* Zoom augmentation!
* Half res support with OpenAIs gradient chekpointing
* Dataset Loaders:
    * Kitti
    * Apolloscapes
    * BDD 100k
    * GTA / VIPER
    * Synthia
    * Virtual Kitti
