# encoding=utf-8
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_prefix')
    parser.add_argument('oup_prefix')
    parser.add_argument('--en-by-zh-ratio', type=float, default=4.0)
    parser.add_argument('--zh-by-en-ratio', type=float, default=2.5)
    parser.add_argument('--min-len', type=int, default=1)
    parser.add_argument('--max-len', type=int, default=175)
    args = parser.parse_args()
    args.src = "en"
    args.tgt = "zh_cn"
    return args


def read(args):
    with open(args.inp_prefix + '.' + args.src, encoding='utf-8') as f:
        src = [x.strip() for x in f]
    with open(args.inp_prefix + '.' + args.tgt, encoding='utf-8') as f:
        tgt = [x.strip() for x in f]
    assert len(src) == len(tgt)
    return src, tgt


def filter_line(s, t, args):
    remove = ("_removed_", "_removed_", 0)
    if s == "_removed_" and t == "_removed_":
        return remove
    ns = len([ss for ss in s.split(' ') if len(ss) > 0])
    nt = len([tt for tt in t.split(' ') if len(tt) > 0])
    if not (args.min_len <= ns < args.max_len):
        return remove
    if not (args.min_len <= nt < args.max_len):
        return remove
    if ns / nt > args.en_by_zh_ratio:
        return remove
    if nt / ns > args.zh_by_en_ratio:
        return remove
    return s, t, 1


def filter_lines(src, tgt, args):
    ret_src = []
    ret_tgt = []
    valid_line = 0
    for s, t in zip(src, tgt):
        s, t, v = filter_line(s, t, args)
        ret_src.append(s)
        ret_tgt.append(t)
        valid_line += v
    return ret_src, ret_tgt, valid_line


def write(src, tgt, args):
    src_fn = args.oup_prefix + '.' + args.src
    tgt_fn = args.oup_prefix + '.' + args.tgt
    # assert not os.path.exists(src_fn)
    # assert not os.path.exists(tgt_fn)
    with open(src_fn, 'w', encoding='utf-8') as f:
        for s in src:
            f.write(s.strip() + '\n')
    with open(tgt_fn, 'w', encoding='utf-8') as f:
        for t in tgt:
            f.write(t.strip() + '\n')


if __name__ == "__main__":
    args = parse_args()
    src, tgt = read(args)
    print("Read %d lines" % len(src))
    src, tgt, valid_line = filter_lines(src, tgt, args)
    print("%d lines after filtering" % valid_line)
    write(src, tgt, args)

