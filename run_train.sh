#!/usr/bin/env bash

cd train_model
python train_trades.py --beta 4.0 --gpu-id GPU-ID
python train_trades.py --beta 6.0 --gpu-id GPU-ID
python train_trades.py --beta 8.0 --gpu-id GPU-ID
