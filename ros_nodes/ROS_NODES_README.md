Here are two ROS Nodes for running the semantic segmentation in a ROS system. You should either put these into your catkin workspace or simply put symlinks to these folders into your catkin.

As far as I know ROS still uses python 2.7. So you will need to either install TensorFlow globally compiled for python 2.7, or create a virtual environment which uses system packages (for ROS) and install tensorflow there. E.g. as such:

```
virtualenv --system-site-packages python27env
source python27ev/bin/activate
pip install -U pip
pip install tensorflow-gpu
```

Then make sure to source your catkin workspace and you are good to go.

# Debugging Image Publisher
This node is only there for debugging. Given a folder it publishes all images of a specified extension in a loop to a specified topic. An actual semantic segmentation node can then subscribe to this topic.

Usage:
```
rosrun debugging_image_publisher publisher.py \
        _image_folder:=<path to images> \
        _fix_width_to:=512 \
        _file_extension:=jpg
```
This will then publish the images in the specified path in a loop while resizing the width to a fixed 512 pixels. Optionally an "fps" can be specified (`_fps`). This determines the rate at which the images will be published, however it considers that loading and resizing takes zero seconds.

# Semantic Segmentation Node
This node performs the actual semantic segmentation. It listens to a topic and feeds the received images through a frozen graph and returns the output in a
colorcoding specific to the dataset the network was trained on. It can be run using the following command:

```
CUDA_VISIBLE_DEVICES=0 rosrun semantic_segmentation_node segmenter.py \
        _input_topic:=<topic to subscribe for input images> \
        _frozen_model:=<path to frozen graph>
```

## Creating a frozen graph
In order to create a frozen graph it is assumed that you have trained a model using the code in the parent folder. In there, there is a `freeze_graph.py` script which can be used to do that. Simply run it using:

```
python freeze_graph.py --experiment_root <path the experiment
```

This will by default pick the last iteration, assuming training has finished (an iteration can be specified) and create a frozen graph in the same experiment_root by appending `_frozen.pb` to the checkpoint filename. Another target can be specified.

# TODO
* Possibly add a configuration where the node returns class indices instead of colors or even probabilities.
* Look into creating fixed input size models as possibly to optimize the inference speed.
