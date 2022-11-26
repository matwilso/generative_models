#!/bin/bash

# Train the teacher original diffusion model, then train a series of progressively distilled models
# usage:
# bash gms/diffusion/progressive_distillation.sh
# or, for cmd dry run
# bash gms/diffusion/progressive_distillation.sh -d

BASE_DIR=logs/$(date +%F)/diffusion_model
BASE_CMD='python -m gms.main --model diffusion_model'

TEACHER_DIR=$BASE_DIR/teacher
STUDENT_DIR=$BASE_DIR/student
MAX_STEPS=256

if [[ $* == *-d* ]]; then
    echo "Dry run. Not executing cmds, just printing them"
    dry_run=1
else
    dry_run=0
fi

# BUILD THE CMDS
# base teacher
cmds=()
cmds+=( "$BASE_CMD --logdir $BASE_DIR/teacher --lr 3e-4 --timesteps $MAX_STEPS" )
# students
# step 1 student, just learns a network that conditions on w, instead of doing classifier free guidance
cmds+=( "$BASE_CMD $STUDENT_ARGS --lr 3e-4 --teacher_path $TEACHER_DIR/model.pt --teacher_mode step1 --logdir $STUDENT_DIR/$MAX_STEPS --timesteps $MAX_STEPS" )
# step 2-N students
STEP2_ARGS="--model diffusion_model --epochs 10 --lr 1e-4 --lr_scheduler none --teacher_mode step2"
previ=$MAX_STEPS
for i in 128 64 32 16 8 4 2 1; do
    cmd="$BASE_CMD $STEP2_ARGS --teacher_path $STUDENT_DIR/$previ/model.pt --logdir $STUDENT_DIR/$i --timesteps $i"
    if [[ $i -lt 16 ]]; then
        # fewer timesteps requires more training since the problem is harder
        cmd="$cmd --epochs 50"
    fi
    cmds+=( "$cmd" )
    previ=$i
done


# PRINT CMDS AND EXECUTE
for cmd in "${cmds[@]}"
do
    if [[ $dry_run == 1 ]]; then
        echo $cmd
    else
        echo $cmd
        eval $cmd
    fi
done
