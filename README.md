# Image Classifier App

The objective of the project is to create a commandline app to identify the flower in an image. A classification model is trained using Tensorflow.

The model training is done using the ipython notebook. The `predict.py`  module  predicts the top flower names from an image along with their corresponding probabilities.

## Basic Usage

```
$ python predict.py /path/to/image saved_model
```

## Options:


--top_k : Return the top K most likely classes:


```
$ python predict.py /path/to/image saved_model --top_k K
```

--category_names : Path to a JSON file mapping labels to flower names:

```
$ python predict.py /path/to/image saved_model --category_names map.json
```
