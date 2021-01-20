#!/usr/bin/env bash

python main_verify.py --model-checkpoint ../checkpoint/resnet50_center.pth
python main_verify.py --model-checkpoint ../checkpoint/resnet50_gce.pth
python main_verify.py --model-checkpoint ../checkpoint/resnet50_pcl.pth

