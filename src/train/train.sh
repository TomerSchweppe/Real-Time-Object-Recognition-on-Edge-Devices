python ./retrain.py \
    --image_dir  ../data\
    --validation_batch_size  -1 \
    --flip_left_right  True \
    --random_crop 10 \
    --random_scale  30 \
    --random_brightness  30 \
    --how_many_training_steps  50000 \
    --tfhub_module  https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2