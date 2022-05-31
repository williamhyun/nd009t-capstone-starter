# Amazon Bin Detector

The purpose of this project is to detect the number of items in a Amazon shipping bin. I plan to offer a different approach to supply chain management using Vision-based Machine Learning to replace RFIDs and barcodes. 

## Dataset

### Overview

The dataset is provided by Amazon, and I have done preprocessing and uploaded to Kaggle to share my work and make other people's projects easier.  
- https://www.kaggle.com/datasets/williamhyun/amazon-bin-image-dataset-536434-images-224x224

The dataset contains 536,434 images of Amazon shipping bins. 

### Access
Kaggle allows for quick and efficient downloads.
Use: `! kaggle datasets download williamhyun/amazon-bin-image-dataset-536434-images-224x224`

## Model Training
I chose to use EfficientNet_b0 since it was the second smallest EfficientNet pre-trained model, which demonstrates great accuracy among all CNNs.
I wanted to use a smaller model since I was on a tight budget. 

To train my EfficientNet_b0 model, I modified the number of epochs, image size, and learning rate and weight decay through optimizer. 
The number of epochs, learning rate, and weight decay are all essential to training a model, however, I chose image size in addition to the essentials to improve the accuracy for detecting empty bins. 

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
I have attempted multi-resolution training and found improvements. 
Multi-resolution training is where one trains a model on a lower resolution version of the dataset and retrains the model with higher resolution images of the same dataset. 
