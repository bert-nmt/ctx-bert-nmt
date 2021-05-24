import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('train_oup')
    parser.add_argument('val_oup')
    parser.add_argument('--startofdoc', required=True)
    parser.add_argument('--valratio', required=True, type=float)
    return parser.parse_args()


def split_and_strip(x, y):
    return [xx.strip() for xx in x.split(y)]


if __name__ == "__main__":
    args = parse_args()

    with open(args.inp) as f:
        text = f.read()
    docs = split_and_strip(text, args.startofdoc)
    assert docs[0] == ""  # startofdoc is the first token
    docs = docs[1:]

    docs = [split_and_strip(x, '\n') for x in docs]
    doc_lens = np.array([len(x) for x in docs], dtype=np.int)
    doc_accums = np.cumsum(doc_lens)

    total = doc_accums[-1]
    split = np.argmin(np.abs(doc_accums - total * (1 - args.valratio))) + 1

    with open(args.train_oup, 'w') as f:
        train_lines = doc_accums[split - 1]
        print("Train: %d documents, %d lines" % (split, train_lines))
        for doc in docs[:split]:
            f.write("%s\n" % args.startofdoc)
            f.write(''.join([x + '\n' for x in doc]))

    with open(args.val_oup, 'w') as f:
        val_lines = total - train_lines
        print("Valid: %d documents, %d lines" % (len(docs) - split, val_lines))
        for doc in docs[split:]:
            f.write("%s\n" % args.startofdoc)
            f.write(''.join([x + '\n' for x in doc]))

    print("Total valid ratio: %.4f" % (val_lines / total))

