import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('--ratio', default=0.2, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # read ids
    ids = [int(x[len(args.inp):]) for x in glob.glob(args.inp + "*")]
    for i in ids:
        # read documents
        with open("{}{:d}".format(args.inp, i), encoding='utf-8') as f:
            lines = [x.strip() for x in f]
        if sum([x == "_removed_" for x in lines]) / len(lines) > args.ratio:
            print(i)

