
TEACHER=logs/1113/final/teacher
python /home/matwilso/code/generative_models/gms/main.py --model diffusion_model --logdir $TEACHER  --epochs 50

STUDENT_A=logs/1113/final/studenta
A_ARGS="--model diffusion_model --epochs 10 --lr 3e-4"
python /home/matwilso/code/generative_models/gms/main.py $A_ARGS --teacher_path $TEACHER --teacher_mode step1 --logdir $STUDENT_A/125 --timesteps 125
python /home/matwilso/code/generative_models/gms/main.py $A_ARGS --teacher_path $STUDENT_A/125 --teacher_mode step2 --logdir $STUDENT_A/64 --timesteps 64
python /home/matwilso/code/generative_models/gms/main.py $A_ARGS --teacher_path $STUDENT_A/64 --teacher_mode step2 --logdir $STUDENT_A/32 --timesteps 32

STUDENT_B=logs/1113/final/studentb
B_ARGS="--model diffusion_model --epochs 10 --lr 1e-4"
python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $TEACHER --teacher_mode step1 --logdir $STUDENT_B/125 --timesteps 125
python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $STUDENT_B/125 --teacher_mode step2 --logdir $STUDENT_B/64 --timesteps 64
python /home/matwilso/code/generative_models/gms/main.py $B_ARGS --teacher_path $STUDENT_B/64 --teacher_mode step2 --logdir $STUDENT_B/32 --timesteps 32

STUDENT_C=logs/1113/final/studentc
C_ARGS="--model diffusion_model --epochs 10 --lr 1e-3"
python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $TEACHER --teacher_mode step1 --logdir $STUDENT_C/125 --timesteps 125
python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $STUDENT_C/125 --teacher_mode step2 --logdir $STUDENT_C/64 --timesteps 64
python /home/matwilso/code/generative_models/gms/main.py $C_ARGS --teacher_path $STUDENT_C/64 --teacher_mode step2 --logdir $STUDENT_C/32 --timesteps 32

