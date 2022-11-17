TEACHER=logs/1114/teacher
#python /home/matwilso/code/generative_models/gms/main.py --model diffusion_model --logdir $TEACHER  --epochs 50

STUDENT=logs/1114/student_c
STUDENT_ARGS="--model diffusion_model --epochs 50"
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 3e-4 --lr_scheduler none    --teacher_path $TEACHER/model.pt --teacher_mode step1 --logdir $STUDENT/256 --timesteps 256
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/256/model.pt --teacher_mode step2 --logdir $STUDENT/128 --timesteps 128
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/128/model.pt --teacher_mode step2 --logdir $STUDENT/64 --timesteps 64
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/64/model.pt --teacher_mode step2 --logdir $STUDENT/32 --timesteps 32
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/32/model.pt --teacher_mode step2 --logdir $STUDENT/16 --timesteps 16
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/16/model.pt --teacher_mode step2 --logdir $STUDENT/8 --timesteps 8
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/8/model.pt --teacher_mode step2 --logdir $STUDENT/4 --timesteps 4
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/4/model.pt --teacher_mode step2 --logdir $STUDENT/2 --timesteps 2
python /home/matwilso/code/generative_models/gms/main.py $STUDENT_ARGS --lr 1e-4 --lr_scheduler linear  --teacher_path $STUDENT/2/model.pt --teacher_mode step2 --logdir $STUDENT/1 --timesteps 1

#STUDENT_B=logs/1113/final3/studentb
#B_ARGS="--model diffusion_model --epochs 10 --lr 1e-4"
#python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $TEACHER/model.pt --teacher_mode step1 --logdir $STUDENT_B/250 --timesteps 250
#python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $STUDENT_B/250/model.pt --teacher_mode step2 --logdir $STUDENT_B/125 --timesteps 125
#python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $STUDENT_B/125/model.pt --teacher_mode step2 --logdir $STUDENT_B/64 --timesteps 64
#python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $STUDENT_B/64/model.pt --teacher_mode step2 --logdir $STUDENT_B/32 --timesteps 32
#
#STUDENT_C=logs/1113/final3/studentc
#C_ARGS="--model diffusion_model --epochs 10 --lr 1e-3"
#python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $TEACHER/model.pt --teacher_mode step1 --logdir $STUDENT_C/250 --timesteps 250
#python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $STUDENT_C/250/model.pt --teacher_mode step2 --logdir $STUDENT_C/125 --timesteps 125
#python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $STUDENT_C/125/model.pt --teacher_mode step2 --logdir $STUDENT_C/64 --timesteps 64
#python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $STUDENT_C/64/model.pt --teacher_mode step2 --logdir $STUDENT_C/32 --timesteps 32
#
#STUDENT_D=logs/1113/final3/studentd
#D_ARGS="--model diffusion_model --epochs 10 --lr 1e-5"
#python /home/matwilso/code/generative_models/gms/main.py $D_ARGS --teacher_path $TEACHER/model.pt --teacher_mode step1 --logdir $STUDENT_D/250 --timesteps 250
#python /home/matwilso/code/generative_models/gms/main.py $D_ARGS --teacher_path $STUDENT_D/250/model.pt --teacher_mode step2 --logdir $STUDENT_D/125 --timesteps 125
#python /home/matwilso/code/generative_models/gms/main.py $D_ARGS --teacher_path $STUDENT_D/125/model.pt --teacher_mode step2 --logdir $STUDENT_D/64 --timesteps 64
#python /home/matwilso/code/generative_models/gms/main.py $D_ARGS --teacher_path $STUDENT_D/64/model.pt --teacher_mode step2 --logdir $STUDENT_D/32 --timesteps 32
#
#STUDENT_E=logs/1113/final3/studentd
#E_ARGS="--model diffusion_model --epochs 10 --lr 5e-5"
#python /home/matwilso/code/generative_models/gms/main.py $E_ARGS --teacher_path $TEACHER/model.pt --teacher_mode step1 --logdir $STUDENT_E/250 --timesteps 250
#python /home/matwilso/code/generative_models/gms/main.py $E_ARGS --teacher_path $STUDENT_E/250/model.pt --teacher_mode step2 --logdir $STUDENT_E/125 --timesteps 125
#python /home/matwilso/code/generative_models/gms/main.py $E_ARGS --teacher_path $STUDENT_E/125/model.pt --teacher_mode step2 --logdir $STUDENT_E/64 --timesteps 64
#python /home/matwilso/code/generative_models/gms/main.py $E_ARGS --teacher_path $STUDENT_E/64/model.pt --teacher_mode step2 --logdir $STUDENT_E/32 --timesteps 32
