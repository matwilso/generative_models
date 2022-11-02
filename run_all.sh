#!/usr/bin/env bash
# usage: bash scripts/run_all.sh NUM_EPOCHS DIRNAME
# defaults are 1, run_all
epochs=${1:-1}
logdir=${2:-run_all}

python gms/main.py --epochs=$epochs --model=rnn --logdir=logs/$logdir/rnn
python gms/main.py --epochs=$epochs --model=made --logdir=logs/$logdir/made
python gms/main.py --epochs=$epochs --model=wavenet --logdir=logs/$logdir/wavenet
python gms/main.py --epochs=$epochs --model=pixelcnn --logdir=logs/$logdir/pixelcnn
python gms/main.py --epochs=$epochs --model=gatedcnn --logdir=logs/$logdir/gatedcnn
python gms/main.py --epochs=$epochs --model=transformer --logdir=logs/$logdir/transformer
python gms/main.py --epochs=$epochs --model=vae --logdir=logs/$logdir/vae
python gms/main.py --epochs=$epochs --model=vqvae --logdir=logs/$logdir/vqvae
python gms/main.py --epochs=$epochs --model=gan --logdir=logs/$logdir/gan
python gms/main.py --epochs=$epochs --model=diffusion --logdir=logs/$logdir/diffusion
