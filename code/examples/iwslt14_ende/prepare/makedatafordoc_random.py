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
size_str = "1random"

for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    tgtfn = '{}{}.bert.doc.{}'.format(subset, size_str, lng)
    print("%s  ==>  %s" % (fn, tgtfn))
    prevs = []

    with open(fn, 'r', encoding='utf8', newline='\n') as f:
        src_lines = [x.strip() for x in f.readlines()]
    src_lines = [x for x in src_lines if x != args.startofdoc]

    idx = np.arange(len(src_lines))
    while np.sum(idx == np.arange(len(src_lines))) != 0:
        np.random.shuffle(idx)
    context = [src_lines[i] for i in idx]

    with open(tgtfn, 'w', encoding='utf8', newline='\n') as tgt:
        for l, cl in zip(src_lines, context):
            newline = '[CLS] {} [SEP] {} [SEP]'.format(cl, l)
            tgt.write(newline + '\n')
    bpein = "{}.{}".format(subset, lng)
    docin = "{}{}.{}.doc.in".format(subset, size_str, lng)
    print("Paste: %s + %s  ==>  %s" % (bpein, tgtfn, docin))
    paste_cmd = "paste -d \"\n\" {} {} > {}".format(bpein, tgtfn, docin)
    print("cmd: %s" % paste_cmd)
    os.system(paste_cmd)

