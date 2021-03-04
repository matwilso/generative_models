#!/usr/bin/env bash
# usage: bash scripts/run_all.sh NUM_EPOCHS DIRNAME
# defaults are 1, run_all
if [ -z "$1" ]; then
  num_epochs=1
else
  num_epochs=$1
fi
if [ -z "$2" ]; then
  logdir=run_all
else
  logdir=$2
fi

python gms/main.py --num_epochs=$num_epochs --model=rnn --logdir=logs/$logdir/rnn
python gms/main.py --num_epochs=$num_epochs --model=made --logdir=logs/$logdir/made
python gms/main.py --num_epochs=$num_epochs --model=wavenet --logdir=logs/$logdir/wavenet
python gms/main.py --num_epochs=$num_epochs --model=pixelcnn --logdir=logs/$logdir/pixelcnn
python gms/main.py --num_epochs=$num_epochs --model=gatedcnn --logdir=logs/$logdir/gatedcnn
python gms/main.py --num_epochs=$num_epochs --model=transformer --logdir=logs/$logdir/transformer
python gms/main.py --num_epochs=$num_epochs --model=vae --logdir=logs/$logdir/vae
python gms/main.py --num_epochs=$num_epochs --model=vqvae --logdir=logs/$logdir/vqvae
python gms/main.py --num_epochs=$num_epochs --model=gan --logdir=logs/$logdir/gan
python gms/main.py --num_epochs=$num_epochs --model=diffusion --logdir=logs/$logdir/diffusion
