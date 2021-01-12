#!/usr/bin/env bash
python3 vae.py --bn=1 --name=logs/bn/
python3 vae.py --bn=0 --name=logs/def/
python3 vae.py --b=0.1 --name=logs/0.1B/
python3 vae.py --bs=1024 --name=logs/1024/
python3 vae.py --z_size=32 --name=logs/32z/
