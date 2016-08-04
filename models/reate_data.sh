# path to the folder, which will contain converted images
EXAMPLE=/home/marta/PycharmProjects/learning_caffe/resource

#path to for_train.txt and for_test.txt
DATA=/home/marta/PycharmProjects/learning_caffe/resource

TRAIN_DATA_ROOT=for_train.txt
VAL_DATA_ROOT=for_test.txt

RESIZE=true     # as not all images are the same size

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/for_test.txt \
    $EXAMPLE/test_lmdb


echo "Done"