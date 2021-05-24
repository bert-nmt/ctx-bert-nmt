import io
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lng', type=str, default='en')
parser.add_argument('--startofdoc', default="startofdocumentplaceholder")
parser.add_argument('--ctx-size', default=1, type=int)
parser.add_argument('--subsets', default=['train', 'valid', 'test', ], nargs='+')
args = parser.parse_args()

lng = args.lng
print('lng {}'.format(lng))
subsets = args.subsets
for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    assert os.path.exists(fn)

print("Using start of doc: %s" % args.startofdoc)
size_str = "{:d}next".format(args.ctx_size)
for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    tgtfn = '{}{}.bert.doc.{}'.format(subset, size_str, lng)
    print("%s  ==>  %s" % (fn, tgtfn))
    prevs = []

    with io.open(fn, 'r', encoding='utf8', newline='\n') as src:
        src_lines = [line.strip() for line in src]

    with io.open(tgtfn, 'w', encoding='utf8', newline='\n') as tgt:
        for i, line in enumerate(src_lines):
            if line != args.startofdoc:
                nexts = src_lines[i + 1:i + args.ctx_size + 1]
                if args.startofdoc in nexts:
                    nexts = nexts[:nexts.index(args.startofdoc)]
                tgt.write('[CLS] {} [SEP] {} [SEP]\n'.format(line, ' '.join(nexts)))
    bpein = "{}.{}".format(subset, lng)
    docin = "{}{}.{}.doc.in".format(subset, size_str, lng)
    print("Paste: %s + %s  ==>  %s" % (bpein, tgtfn, docin))
    paste_cmd = "paste -d \"\n\" {} {} > {}".format(bpein, tgtfn, docin)
    print("cmd: %s" % paste_cmd)
    os.system(paste_cmd)

