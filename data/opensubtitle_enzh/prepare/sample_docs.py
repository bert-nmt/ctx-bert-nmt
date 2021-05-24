import argparse
import glob
import os
from random import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('--src')
    parser.add_argument('--tgt')
    parser.add_argument('--exclude', nargs="*")
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    return args


def read_ids(x):
    with open(x, encoding='utf-8') as f:
        return [int(xx.strip()) for xx in f]


def write_ids(x, ids):
    with open(x, 'w', encoding='utf-8') as f:
        for i in ids:
            f.write("{:d}\n".format(i))


def glob_ids(args):
    inp_src = "{}.{}.doc_*".format(args.inp, args.src)
    inp_tgt = "{}.{}.doc_*".format(args.inp, args.tgt)
    ids = [int(x[len(inp_src) - 1:]) for x in glob.glob(inp_src)]
    assert set(ids) == set([int(x[len(inp_tgt) - 1:]) for x in glob.glob(inp_tgt)])
    return ids


if __name__ == "__main__":
    args = parse_args()

    # read ids
    ids = glob_ids(args)
    # exclude ids
    exclude_ids = set()
    for ex in args.exclude:
        exclude_ids = exclude_ids.union(set(read_ids(ex)))
    ids = sorted(set(ids) - exclude_ids)
    print("%d documents can be used" % len(ids))

    # select and write documents
    shuffle(ids)
    n_lines = 0
    src_out = args.out + "." + args.src + ".with_startofdoc"
    tgt_out = args.out + "." + args.tgt + ".with_startofdoc"
    with open(src_out, 'w', encoding='utf-8') as fsrc:
        with open(tgt_out, 'w', encoding='utf-8') as ftgt:
            # iterate ids
            for n_doc, i in enumerate(ids):
                # check stopping or not
                print("Write to %s and %s: %d / %d lines" % (src_out, tgt_out, n_lines, args.n), end='\r')
                if n_lines > args.n:
                    break
                # read documents
                with open("{}.{}.doc_{:d}".format(args.inp, args.src, i), encoding='utf-8') as f:
                    src_lines = [x.strip() for x in f]
                with open("{}.{}.doc_{:d}".format(args.inp, args.tgt, i), encoding='utf-8') as f:
                    tgt_lines = [x.strip() for x in f]
                assert len(src_lines) == len(tgt_lines)
                # write
                fsrc.write("startofdocumentplaceholder\n")
                for l in src_lines:
                    fsrc.write(l + '\n')
                ftgt.write("startofdocumentplaceholder\n")
                for l in tgt_lines:
                    ftgt.write(l + '\n')
                n_lines += len(src_lines)
    print("\nTotal: use %d documents, get %d lines" % (n_doc, n_lines))
    write_ids(args.out + ".ids", ids[:n_doc])

