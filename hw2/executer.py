from os import system


LS = [2, 4, 8, 16]
KS = [32, 64]
H = 100

system('clear')
for k in KS:
    for l in LS:
        p = 4 if l >=8 else 2
        h = H
        name = f'exp1_1'
        command = f'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n {name} -K {k} -L {l} -P {p} -H {h}'
        system(command)