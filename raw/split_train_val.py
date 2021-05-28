import os
import random
import sys

if __name__ == '__main__':
    seed = sys.argv[1]
    random.seed(seed)
    filename = 'train.csv'
    with open(filename) as fh:
        data = fh.readlines()
    data = data[1:]
    print(f'LOAD from {filename}, #ins: {len(data)}')
    random.shuffle(data)
    print(f'SHUFFLED, seed: {seed}')

    dir_ = f'seed-{seed}'
    os.makedirs(dir_)
    filename = os.path.join(dir_, 'train.csv')
    with open(filename, 'w') as fh:
        for index in range(4000):
            fh.write(data[index])
    print(f'OUTPUT to {filename}, #train-ins: 4000')

    filename = os.path.join(dir_, 'val.csv')
    with open(filename, 'w') as fh:
        for index in range(1000):
            fh.write(data[index + 4000])
    print(f'OUTPUT to {filename}, #val-ins: 1000')
