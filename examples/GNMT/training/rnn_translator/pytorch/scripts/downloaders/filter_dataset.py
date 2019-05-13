import argparse
import string
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description='Clean dataset')
    parser.add_argument('-f1', '--file1', help='file1')
    parser.add_argument('-f2', '--file2', help='file2')
    return parser.parse_args()


def save_output(fname, data):
    with open(fname, 'w') as f:
        f.writelines(data)

def main():
    args = parse_args()

    c = Counter()
    skipped = 0
    valid = 0
    data1 = []
    data2 = []

    with open(args.file1) as f1, open(args.file2) as f2:
        for idx, lines in enumerate(zip(f1, f2)):
            line1, line2 = lines
            if idx % 100000 == 1:
                print('Processed {} lines'.format(idx))
            try:
                line1.encode('latin1')
                line2.encode('latin1')
            except UnicodeEncodeError:
                skipped += 1
            else:
                data1.append(line1)
                data2.append(line2)
                valid += 1
                c.update(line1)

    ratio = valid / (skipped + valid)
    print('Skipped: {}, Valid: {}, Valid ratio {}'.format(skipped, valid, ratio))
    print('Character frequency:', c)

    save_output(args.file1, data1)
    save_output(args.file2, data2)


if __name__ == '__main__':
    main()
