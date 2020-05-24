############## To train the model with new line detectors #############
python train.py \
    --name full_line_10 \
    --dataroot ./dataset/train/ \
    --label_nc 0 \
    --gpu_ids 0 \
    --batchSize 16 \
    --loadSize 256 \
    --fineSize 256 \
    --no_instance \
    --lambda_prior 10 \
    --continue_train \
    --detector_type line \
    --tf_log