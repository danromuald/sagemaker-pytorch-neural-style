#!/usr/bin bash

# Download VGG pre-trained model on the 21styles-style images.
# Notes:
#
# Mbanga, 2017
#
# var
idVer=1.0

PROJECT_HOME=`pwd`

function download_vgg () {
    cd models
    wget -O 21styles.params https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/21styles-32f7205c5.params
}

# Main

download_vgg

exit 0