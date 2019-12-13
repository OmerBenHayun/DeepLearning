from os import system
import sys

def run_exp(name , K, L, model='cnn'):
    H = 100
    LS = L
    KS = K
    system('clear')
    for k in KS:
        for l in LS:
            num_of_layers = (str(k).count(' ') + 1) * l
            p = num_of_layers // 3 + 1
            h = H
            command = f'srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n {name} -K {k} -L {l} -P {p} -H {h} -M {model}'
            system(command)


if sys.argv[1] == '1':
    run_exp('exp1_1', [32, 64], [2, 4, 8, 16])
elif sys.argv[1] == '2':
    run_exp('exp1_2', [32, 64, 128, 256], [2, 4, 8])
elif sys.argv[1] == '3':
    run_exp('exp1_3', ['64 128 256'], [1, 2, 3, 4])
elif sys.argv[1] == '4':
    run_exp('exp1_4', [32, '64 128 256'], [2, 4, 8, 16, 32], 'resnet')
    system('rm results/exp1_4_L2_K32.json')
    system('rm results/exp1_4_L4_K32.json')
    system('rm results/exp1_4_L16_K64-128-256.json')
    system('rm results/exp1_4_L32_K64-128-256.json')
