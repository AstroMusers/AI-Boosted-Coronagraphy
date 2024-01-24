import subprocess


########################################################################
########################################################################
######## SIMULTANEOUS EXPERIMENT script
########################################################################
########################################################################

exp_h = [
    "python train_ae.py --device cuda:0 --idx ae_filter/4 --wandb --loss_type l1 --seed 0 --lr 3e-5 --batch_size 256 --model ae --scheduler --apply_lowpass",
    "python train_ae.py --device cuda:0 --idx ae_filter/5 --wandb --loss_type l1 --seed 0 --lr 3e-4 --batch_size 256 --model ae --scheduler --apply_lowpass",
    "python train_ae.py --device cuda:0 --idx ae_filter/6 --wandb --loss_type l1 --seed 0 --lr 3e-3 --batch_size 256 --model ae --scheduler --apply_lowpass",


    "python train_ae.py --device cuda:3 --idx ae_filter/7 --wandb --loss_type l2 --seed 0 --lr 3e-4 --batch_size 512 --model ae  --scheduler --apply_lowpass --latent_dim 8", #
    "python train_ae.py --device cuda:3 --idx ae_filter/8 --wandb --loss_type l2 --seed 0 --lr 3e-4 --batch_size 512 --model ae  --scheduler --apply_lowpass --latent_dim 16", #
    "python train_ae.py --device cuda:3 --idx ae_filter/9 --wandb --loss_type l2 --seed 0 --lr 3e-4 --batch_size 512 --model ae  --scheduler --apply_lowpass --latent_dim 4", #
    "python train_ae.py --device cuda:3 --idx ae_filter/10 --wandb --loss_type l2 --seed 0 --lr 3e-4 --batch_size 512 --model ae  --scheduler --apply_lowpass --latent_dim 2", #
]


def main():
    
    bashcode = ''
    python_commands = exp_h

    commands = {
        f"new-{i}":bashcode + el for i,el in enumerate(python_commands)
    }
    
    eval = 'eval "$(conda shell.bash hook)"'
    for k,v in commands.items():
        
        code1 = f"tmux+new-session+-d+-s+{k}"
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        code3 = f"tmux+send-keys+-t+{k}+conda activate jwst-dev+Enter"
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        for i in [code1, code2, code3, code4]:
            res = subprocess.run(i.split('+'))
            print(res)


if __name__ == '__main__':
    main()
