# usage: bash scripts/run_all.sh
python gms/main.py --num_epochs=1 --model=rnn --logdir=logs/run_all/rnn
python gms/main.py --num_epochs=1 --model=made --logdir=logs/run_all/made
python gms/main.py --num_epochs=1 --model=wavenet --logdir=logs/run_all/wavenet
python gms/main.py --num_epochs=1 --model=pixelcnn --logdir=logs/run_all/pixelcnn
python gms/main.py --num_epochs=1 --model=gatedcnn --logdir=logs/run_all/gatedcnn
python gms/main.py --num_epochs=1 --model=transformer --logdir=logs/run_all/transformer
python gms/main.py --num_epochs=1 --model=vae --logdir=logs/run_all/vae
python gms/main.py --num_epochs=1 --model=vqvae --logdir=logs/run_all/vqvae
python gms/main.py --num_epochs=1 --model=gan --logdir=logs/run_all/gan