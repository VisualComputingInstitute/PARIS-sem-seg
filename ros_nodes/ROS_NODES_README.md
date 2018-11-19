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
        _image_topic:=<topic to subscribe for input images> \
        _frozen_model:=<path to frozen graph>
```

TensorRT is integrated into the node and can be applied to the frozen model that is loaded by speficying a TensorRT mode. Currently `FP16` and `FP32` are supported. There is no guarantee that this will work, for one the results might actually break if for example to optimization results into over/underflows of a `float16` or if the optimization simply fails becomes some ops in the frozen model cannot be optimized. One of my favorites (the `concat` op) for example does not work and I had to change the models slightly to get TensorRT to work.

## Running the Nodes on a Jetson TX2
When installed through JetPack, the performance of a Jetson can be further optimized by running the following commands:
```
./ sudo jetson_clocks.sh
sudo nvpmodel -m 0
```
As far as I recall, the first will set the clocks to be at max always and the second increases the max values. In practice, this can easily give a 10% performance bump.

It should also be noted that the Jetson TX2 is a very limited board, meaning you can run the image publisher on it at the same time, but this will eat up computational power and thus reduce the speed at which the model can run. Another option is to run the image publisher node (or possibly the camera node) on another machine. The Jetson can be connected via ethernet, although here is is important to correctly specify the IP addresses in the `/etc/hosts` file and to specify the `ROS_HOSTNAME`, which simply reflects the hostname and the `ROS_MASTER_URI` which specifies where the `roscore` will run, on both machines. Likely this should go into some start up script or even into a `~/.bashrc`. The `ROS_MASTER_URI` has the following format:
```
export ROS_MASTER_URI=http://<roscore_host_name>:11311
```
Where the port can likely be replaced by something else, as long as it is consistent across machines.
## Creating a frozen graph
In order to create a frozen graph it is assumed that you have trained a model using the code in the parent folder. In there, there is a `freeze_graph.py` script which can be used to do that. Simply run it using:

```
python freeze_graph.py --experiment_root <path the experiment>
```

This will by default pick the last iteration, assuming training has finished (an iteration can be specified) and create a frozen graph in the same experiment_root by appending `_frozen.pb` to the checkpoint filename. Another target can be specified. If this graph should be optimized based on TensorRT, it is important to also fix the input dimensions. This has the big disadvantage that the model can only be applied to a single specific image size, but it can give huge performance boosts during runtime. To do so, also specify the following parameters: `--fixed_input_height`, `--fixed_input_width`, and `--fixed_batch_size`.
