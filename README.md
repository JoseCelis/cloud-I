# cloud-I

Blog post explaining this project in more detail can be found [HERE](https://medium.com/p/b553e6576af6/edit).

## Description

Set of ML algorithms for cloud detection in stellite images trained using Sentinel-2 imagery. 

In this project we download and preprocess image files from the data collection SENTINEL2_L2A
using sentinelhub.

## Import images
Downloads the images from remote repository using sentinelhub python package.
Be sure you already requested your CLIENT_ID and CLIENT_SECRET.

Once you have the credentials include them in a .env file stored in the root folder of
this project, as it is shown here.

```commandline
# sentinelhub credentials
CLIENT_ID = "********************************"
CLIENT_SECRET = "*******************************"
```

## Preprocess
Improves the brightness and contrast of the images using two transformations and creates cloud masks using the cloud probability file downloaded.
