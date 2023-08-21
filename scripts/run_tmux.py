import subprocess


exp_h = [
    "python train.py --device cuda:0 --idx injection_ae/0 --wandb --seed 0 --lr 0.001 --batch_size 256",
    "python train.py --device cuda:1 --idx injection_ae/1 --wandb --seed 0 --lr 3e-4 --batch_size 256",
]



def main():
    
    bashcode = ''
    python_commands = exp_h

    commands = {
        f"sess-{i+1}":bashcode + el for i,el in enumerate(python_commands)
    }
    
    eval = 'eval "$(conda shell.bash hook)"'
    for k,v in commands.items():
        
        code1 = f"tmux+new-session+-d+-s+{k}"
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        code3 = f"tmux+send-keys+-t+{k}+conda activate jwst-dev+Enter"
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        for i in [code1, code2, code3,code4]:
            res = subprocess.run(i.split('+'))
            print(res)


if __name__ == '__main__':
    main()
