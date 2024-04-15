# cloud-I

Blog post explaining this project in more detail can be found [HERE](https://medium.com/p/b553e6576af6/edit).

## Description

Set of ML algorithms for cloud detection in satellite images trained using Sentinel-2 imagery. 

In this project we download and preprocess image files from the data collection SENTINEL2_L2A
using sentinelhub.

## TODO: 
[Add prerrequisites needed to run this repo]


### example .env
In the root folder, create a .env file with the following information:
```commandline
# sentinelhub credentials
CLIENT_ID = "********************************"
CLIENT_SECRET = "*******************************"

# dagshub ML_FLOW credentials
MLFLOW_TRACKING_URI="*******************************"
MLFLOW_TRACKING_USERNAME="********"
MLFLOW_TRACKING_PASSWORD="************************************"
```
[Write here how to get the sentinelhub credentials and the dagshub credentials].


## Import images
[Mention the Python script used for this]

Downloads the images from remote repository using sentinelhub python package [Add link].
Be sure you already requested your CLIENT_ID and CLIENT_SECRET  [in this link, add the link].

Once you have the credentials, include them in a .env file stored in the root folder of
this project, as it is shown iin the example above.

## Preprocess
[Mention the Python script used for this]

Improves the brightness and contrast of the images using two transformations and creates cloud masks using the cloud 
probability file downloaded.

Data augmentation is done by rotating the pictures 90, 180 and 270 degrees, as well as vertical and horizontal flips.

## Model
[Mention the Python script used for this]

Different models are implemeted: Random Forest, ANN, FCNN, UNET.

