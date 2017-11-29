#!/usr/bin bash

# Download COCO dataset
# Notes:
#
#
#
# var
idVer=1.0

PROJECT_HOME=`pwd`

function download_coco () {
    cd /opt/ml/input/data
    wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    unzip train2014.zip
    unzip val2014.zip
    rm *.zip
    rm -rf train/
    rm -rf eval/
    mv train2014 train
    mv val2014 eval
}

# Main

download_coco

exit 0
