Introduction
============

This project purpose to classify images on custom data. It's a ready-to-use project to train models and pre-trained models to make inferences on custom data. In addition, It allows to do data augmentation.

Installation
============

    conda create --name env_image_classification python=3.7
    conda config --add channels conda-forge
    conda install tensorflow matplotlib pandas numpy scikit-learn opencv tqdm imgaug
    conda install -c anaconda absl-py 
    pip install tensorflow-addons

(Optional) Run the test file with pytest. 
(Optional) install tensorflow-gpu

Prepare your data
=================

You need a folder with your images and a .csv file with 2 columns ('id' (name of your images) and 'targets'). To create the .csv file, you can use label_extractor.py.

Training and Inferences
=======================

Some examples on how to use the project, see more details in Command Line Args Reference sectrion.

### Training and inferences

    python image_classification.run.py --mode all -num_epochs 50

### Training only

    python image_classification.run.py --mode train --train_img_folder data/train/ --train_targets data/train.csv -num_epochs 50

### Inferences only

    python image_classification.run.py --mode infer --weights weights_vgg16_epoch_30.h5 --infer_img_folder data/test/


Data Augmentation
=================

[imgaug library](https://github.com/aleju/imgaug) is implemented to do data augmentation.  To use it, you just have to choose a percentage of data augmentation you want (compared to your data's number) and it will create and integrate the new data to your data and train the model. The pipeline to increase the data could change depending on your data. To change the pipeline, you can change the seq variable in data_preparation_ic.py in data_augmentation function.

For example:

    seq = iaa.Sequential([
        iaa.Crop(px=(1, 16), keep_size=False),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])

See all you can do in the [imgaug documentation](https://imgaug.readthedocs.io/en/latest/).

Command Line Args Reference
===========================

    image_classification.run.py:
    --mode: train, infer, evaluate or all (train, infer and evaluate)
        (default: 'all')
    --classifier: vgg16 or mobileNetv2 or TripletLoss
        (default: 'vgg16')
    --train_img_folder: path to image folder to train
        (default: 'data/train/')
    --train_targets: path to targets file
        (default: 'data/train.csv')
    --weights: path to weights file to load (.h5)
        (default: None)
    --infer_img_folder: path to image folder to test
        (default: 'data/test/')
    --infer_img: path to image file to infer
        (default: None)
    --BATCH_SIZE: batch size
        (default: 64)
        (an integer)
    --num_epochs: number of epochs
        (default: 30)
        (an integer)
    --IMAGE_SIZE: image size depending on the model
        (default: 32)
        (an integer)
    --val_step: number of validation step
        (default: 20)
        (an integer)
    --img_aug: Proceed data augmentation, 0 means no data augmentation, 100 means 100% of the data
        (default: 0)
        (an integer)
    --display_data: Display 25 data and labels randomly
        (default: False)
        (an boolean)
    --lite: convert, save and infer to tensorflow Lite
    (default: False)
    (an boolean)
    --lite_output: output path to save .tflite model file (weight)
    (default: 'model_quantized_classifier.tflite')
        
Performance on Kaggle cactus dataset
====================================

|                                    | TensorFlow-CPU | TensorFlow-GPU | TFlite |
|------------------------------------|----------------|----------------|--------|
| Accuracy on validation (50 epochs) | 0.997          | 0.998          |        |
| inference Time (ms)                | 12.76          | 4.17           | 321    |

Results with vgg16 based model.

Test results
============

collected 8 items
test_image_classification.py::Testdata_preparation::test_data_proc PASSED                                [ 12%] 
test_image_classification.py::Testdata_preparation::test__pre_proc PASSED                                [ 25%]
test_image_classification.py::Testdata_preparation::test_data_augmentation PASSED                        [ 37%]
test_image_classification.py::Testmodel::test_vgg16_model PASSED                                         [ 50%]
test_image_classification.py::Testmodel::test_MobileNetV2_model PASSED                                   [ 62%]
test_image_classification.py::TestImage_classification::test_tensorflow_installation PASSED              [ 75%]
test_image_classification.py::TestImage_classification::test_infer PASSED                                [ 87%]
test_image_classification.py::TestImage_classification::test_train PASSED                                [100%]

**----------- coverage: platform linux, python 3.7.7-final-0 -----------**

| Name                                      | Stmts | Miss | Cover |
|-------------------------------------------|-------|------|-------|
| image_classification.py                   | 114   | 63   | 45%   |
| modules_ic/data_preparation_ic.py         | 51    | 19   | 63%   |
| modules_ic/model_ic.py                    | 20    | 0    | 100%  |
| run_image_classification.py               | 46    | 46   | 0%    |
| test_image_classification.py              | 69    | 0    | 100%  |
| TOTAL                                     | 300   | 128  | 57%   |


