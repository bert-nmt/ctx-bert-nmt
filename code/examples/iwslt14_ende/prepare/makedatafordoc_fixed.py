import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('lng', type=str, default='en')
parser.add_argument('--startofdoc', default="startofdocumentplaceholder")
args = parser.parse_args()

lng = args.lng
print('lng {}'.format(lng))
subsets = ['train', 'valid', 'test']
for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    assert os.path.exists(fn)

print("Using start of doc: %s" % args.startofdoc)
size_str = "1fix"

fix = None

for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    tgtfn = '{}{}.bert.doc.{}'.format(subset, size_str, lng)
    print("%s  ==>  %s" % (fn, tgtfn))
    prevs = []

    with open(fn, 'r', encoding='utf8', newline='\n') as f:
        src_lines = [x.strip() for x in f.readlines()]
    src_lines = [x for x in src_lines if x != args.startofdoc]

    if fix is None:
        assert subset == 'train'
        fix = np.random.choice(src_lines)

    with open(tgtfn, 'w', encoding='utf8', newline='\n') as tgt:
        for l in src_lines:
            newline = '[CLS] {} [SEP] {} [SEP]'.format(fix, l)
            tgt.write(newline + '\n')
    bpein = "{}.{}".format(subset, lng)
    docin = "{}{}.{}.doc.in".format(subset, size_str, lng)
    print("Paste: %s + %s  ==>  %s" % (bpein, tgtfn, docin))
    paste_cmd = "paste -d \"\n\" {} {} > {}".format(bpein, tgtfn, docin)
    print("cmd: %s" % paste_cmd)
    os.system(paste_cmd)

