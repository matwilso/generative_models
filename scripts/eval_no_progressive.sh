python -m gms.main --model diffusion_model --epochs 0 --weights_from logs/1114/student_a/256/model.pt --logdir logs/1114/student_a/256_eval/128 --timesteps 128 --skip_train 1 --save_n 1
python -m gms.main --model diffusion_model --epochs 0 --weights_from logs/1114/student_a/256/model.pt --logdir logs/1114/student_a/256_eval/64 --timesteps 64 --skip_train 1 --save_n 1
python -m gms.main --model diffusion_model --epochs 0 --weights_from logs/1114/student_a/256/model.pt --logdir logs/1114/student_a/256_eval/32 --timesteps 32 --skip_train 1 --save_n 1
python -m gms.main --model diffusion_model --epochs 0 --weights_from logs/1114/student_a/256/model.pt --logdir logs/1114/student_a/256_eval/16 --timesteps 16 --skip_train 1 --save_n 1
python -m gms.main --model diffusion_model --epochs 0 --weights_from logs/1114/student_a/256/model.pt --logdir logs/1114/student_a/256_eval/8 --timesteps 8 --skip_train 1 --save_n 1
