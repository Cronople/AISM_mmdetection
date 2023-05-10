cd /opt/ml/detection/baseline/mmdetection/
python tools/train.py ../../../_boost_/youngun/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py \
--work-dir ./outputs/faster-rcnn-resneXt-higherLr-512