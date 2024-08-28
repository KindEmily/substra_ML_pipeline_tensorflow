# Introduction  
This repo is showcasing the first attempt to adapt Substra example tutorial [Using Torch FedAvg on MNIST dataset](https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html) 

- Original implementation: This is a classification problem aiming to recognize the number written on each image.
- Desirable implementation: Classification problem aiming to train a new model to recognize the image based on dataset from Kaggle [vinbigdata-chest-xray-abnormalities-detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

# Setup env 

```
# Open repo
cd C:\Users\probl\Work\Substra_env\substra_ML_pipeline_tensorflow

# Create an env 
conda env create -f substra-tensorflow-env.yml

# Activate env 
conda activate ml_substra_tensorflow

# make sure you have train images 

# run pipeline 
python run_pipeline.py (if you have images downloaded)

# Get error (see above or folder `substra_pipeline_tensorflow\substra_ML_pipeline_tensorflow\logs`)
```

# Dataset 
Download images from the [VinBigData Chest X-ray Abnormalities Detection ](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data?select=train) dataset 

