#!/bin/bash

python test.py --name full_line_10 --dataroot ./dataset/test/cross_floor \
  --label_nc 0 --gpu_ids 1 --batchSize 4 --loadSize 256 --fineSize 256 --no_instance --which_epoch best


