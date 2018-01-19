# Semantic Segmentation for PARIS

# Design space
* Convolution type
    * normal conv
    * depthwise seperable
    * bottle neck resblocks.
* depth multiplier
* Resblock count
* Pooling counts



# TODO
1. Augmentations
2. Eval code based on results in exp dir and images from some json, creating predictions and numbers.
3. Test all three losses
4. Add parametrization to the network (which convs, which blocks etc.)
5. Add an E-net




# Usage

## Dataset Generation
First downscale the images for realistic training. Be aware this will be resizing a lot of images. Since resizing of labels was not implemented super efficiently this will take several hours, but it is less wasteful than doing it on the fly during training.

Example call:
```
python resize_cityscapes.py --downscale_factor 2 \
        --cityscapes_root /work/hermans/cityscapes \
        --target_root /work/hermans/cityscapes/down2
```

