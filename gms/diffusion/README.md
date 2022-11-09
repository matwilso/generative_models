for more details about the implementation, i'm writing up a post on that.


for reference, see the unixpickle and google research implementations.
while it borrows a lot, mine is pretty heavily modified from these



python -m gms.main --model diffusion_model --logdir logs/1106/diffusion_student_weighttest_w0.0 --weights_from logs/1106/diffusion_teachera/model.jit.pt --skip_train 1 --cf_cond_w 0.0 --save_n 1


Experiments to try out:
- trying to generate
- messing with weight.