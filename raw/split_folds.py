import os
import sys


if __name__ == '__main__':
    fold = int(sys.argv[1])
    filename = 'train.csv'
    with open(filename) as fh:
        data = fh.readlines()
    data = data[1:]
    print(f'LOAD from {filename}, #ins: {len(data)}')

    dir_ = f'folds-{fold}'
    os.makedirs(dir_)
    step = len(data) // fold
    for index in range(fold):
        filename = os.path.join(dir_, f'{index+1}.csv')
        with open(filename, 'w') as fh:
            for line_no in range(index*step, (index+1)*step):
                fh.write(data[line_no].strip() + '\n')
        print(f'OUTPUT to {filename}, fold-{index+1} [{index*step}, {(index+1)*step})')

