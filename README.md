# cloud-I

Blog post explaining this project in more detail can be found [HERE](https://medium.com/p/b553e6576af6/edit).

## Description

Set of ML algorithms for cloud detection in satellite images trained using Sentinel-2 imagery. 

In this project we download and preprocess image files from the data collection SENTINEL2_L2A
using sentinelhub.

## Prerequisites
Make sure you have the following software installed:

* Python 3.11

To create a virtual environment of this project, open a terminal, go to the root path of
this project and type this:
```commandline
> pip install pipenv
> pipenv install --dev
```

## Setting up enviromental variables
In the root folder, create a .env file with the following information:
```commandline
# sentinelhub credentials
CLIENT_ID = <your client_id>
CLIENT_SECRET = <your client secret>

# dagshub ML_FLOW credentials
MLFLOW_TRACKING_URI=https://dagshub.com/<your dagshub username>/cloud-I.mlflow
MLFLOW_TRACKING_USERNAME=<your dagshub username>
MLFLOW_TRACKING_PASSWORD=<your mlflow tracking password>
```
To get the sentinelhub credentials:

- Create an account in the [Sentinel Hubs dashboard](https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/auth?client_id=30cf1d69-af7e-4f3a-997d-0643d660a478&redirect_uri=https%3A%2F%2Fapps.sentinel-hub.com%2Fdashboard%2F&state=cd274940-99ee-4c57-8418-f82540051357&response_mode=fragment&response_type=code&scope=openid&nonce=fa1f4d93-8730-49d7-beab-dbe2fc822833&code_challenge=tP9ehp6dDZnaVjnvnJi2DhSAAn0sAkZqvMmAFo1atJ0&code_challenge_method=S256).
- Get an OAuth client. Details [here](https://docs.sentinel-hub.com/api/latest/api/overview/authentication/#registering-oauth-client).

To get the dagshub ML_FLOW credentials, follow [this documentation](https://dagshub.com/DagsHub-Official/dagshub-docs/src/main/docs/integration_guide/mlflow_tracking.md#3-set-up-your-credentials). 


## Import images
You can use the script ```src/import_image.py ```.

This script downloads the images from remote repository using the [sentinelhub python package](https://sentinelhub-py.readthedocs.io/en/latest/).
Be sure you already requested your CLIENT_ID and CLIENT_SECRET.

## Preprocess
To preprocess the Sentinel-2 images, you can use the script ```src/preprocess.py ```.

This script improves the brightness and contrast of the images using two transformations. It also creates cloud masks using the cloud 
probability files downloaded.

Data augmentation is done by rotating the pictures 90, 180 and 270 degrees, as well as vertical and horizontal flips.

## Modeling
You can use the script ```src/model.py ```.

Different models are implemeted: Random Forest, ANN, FCNN, UNET.

To train a specific model, write in your terminal (inside your environment):
```python src/model.py --model_name=<model name>```, where 

* ```--model_name=rf``` trains a Random Forest.
* ```--model_name=ann``` trains an ANN.
* ```--model_name=unet``` trains a U-NET model.
* ```--model_name=segnet``` trains a SegNet model.
* ```--model_name=yolo``` trains YOLO.

If you want to make a prediction with an already trained model that is saved, use:
```python src/model.py --model_name=<model name> --train_model=False```

## FAQ

### I have my own data. Where do I store it to run the models? 
You need to create a folder called ```Dataset/``` in the root path of the project. The structure
of this folder is as follows:
```
├── Dataset                      
│   ├── images  
│       ├── train
│           ├── <train image name 1>.png
│           ├── ...
│       ├── val
│           ├── <validation image name 1>.png
│           ├── ...
│   ├── masks  
│       ├── train
│           ├── <train mask name 1>.png
│           ├── ...
│       ├── val
│           ├── <validation mask name 1>.png
│           ├── ...
│   ├── labels  
│       ├── train
│           ├── <train image name 1>.txt
│           ├── ...
│       ├── val
│           ├── <validation image name 1>.txt
│           ├── ...
│   ├── test
│       ├── <test image name>.png
│       ├── <test mask name>.png
```
***IMPORTANT: The code is designed to test the models on a single image.***

