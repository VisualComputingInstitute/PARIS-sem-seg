# Semantic Segmentation in PARIS

# Usage
* Get data
* convert
* train
* evaluate

# Design space
* Convolution type
* depth multiplier
* Resblock count


# TODO
1. Raw dataset conversion, possibly one dataset class per dataset. With downsampling options. Maybe everything based on json files ?
2. Train stub with resume and parameter dumping
3. Predict that takes a model and creates the images.
4. Eval code based on results in exp dir and images from some json.
