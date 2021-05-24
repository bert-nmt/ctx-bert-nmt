# encoding=utf-8
import re
import argparse
import tqdm
from difflib import SequenceMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_prefix')
    parser.add_argument('oup_prefix')
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


def clean_line(x, p):
    e = 0
    ret = ""
    for m in re.finditer(p, x):
        ret += x[e:m.start()]
        e = m.end()
    return ret + x[e:]


def filter_line(s, t):
    # clean line first
    for p in [r"\{[^\{\}]*\}", r"\[[^\{\}]*\]", r"\{[0-9a-zA-Z\(\)\\:]*", r"[0-9a-zA-Z\(\)\\:]*}"]:
        # {}: subtitle control tokens
        # []: explanations usually
        s = clean_line(s, p)
        t = clean_line(t, p)
    # remove when s == t
    if s == t:
        return "_removed_", "_removed_", 0
    # remove s from t when t contains s (i.e. subtitle with two languages)
    m = SequenceMatcher(None, s, t).find_longest_match(0, len(s), 0, len(t))
    if m.size > 30:
        t = t[:m.b] + t[m.b + m.size:]
    # remove when s contains no English character
    if re.search(r"[a-zA-Z0-9]", s) is None:
        return "_removed_", "_removed_", 0
    # remove when t contains no Chinese character
    if re.search(u'[\u4e00-\u9fff]', t) is None:
        return "_removed_", "_removed_", 0
    # remove if s or t is empty
    if len(s) == 0 or len(t) == 0:
        return "_removed_", "_removed_", 0
    return s, t, 1


def filter_lines(src, tgt, args):
    ret_src = []
    ret_tgt = []
    valid_line = 0

    for s, t in tqdm.tqdm(list(zip(src, tgt))):
        s, t, v = filter_line(s, t)
        ret_src.append(s)
        ret_tgt.append(t)
        valid_line += v
    return ret_src, ret_tgt, valid_line


def write(src, tgt, args):
    src_fn = args.oup_prefix + '.' + args.src
    tgt_fn = args.oup_prefix + '.' + args.tgt
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
