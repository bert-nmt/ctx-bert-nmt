import io
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lng', type=str, default='en')
parser.add_argument('--startofdoc', default="startofdocumentplaceholder")
# parser.add_argument('--ctx-size', default=1, type=int)
args = parser.parse_args()

lng = args.lng
print('lng {}'.format(lng))
subsets = ['train', 'valid', 'test']
for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    assert os.path.exists(fn)

print("Using start of doc: %s" % args.startofdoc)
size_str = "currnext"  # "" if args.ctx_size == 1 else "{:d}".format(args.ctx_size)
for subset in subsets:
    fn = '{}.bert.{}'.format(subset, lng)
    tgtfn = '{}{}.bert.doc.{}'.format(subset, size_str, lng)
    print("%s  ==>  %s" % (fn, tgtfn))
    prev = ""  # prevs = []
    with io.open(fn, 'r', encoding='utf8', newline='\n') as src:
        with io.open(tgtfn, 'w', encoding='utf8', newline='\n') as tgt:

            def print_sents(prev, curr):
                newline = '[CLS] {} [SEP] {} [SEP]'.format(prev, curr)
                tgt.write(newline + '\n')

            for line in src:
                line = line.strip()
                if line:
                    if line == args.startofdoc:
                        line = ""  # prevs = []
                    # else:
                    #     newline = '[CLS] {} [SEP] {} [SEP]'.format(' '.join(prevs), line)
                    #     prevs = prevs[len(prevs) - args.ctx_size + 1:] + [line, ]
                    #     tgt.write(newline + '\n')
                    if len(prev) > 0:
                        print_sents(prev, line)
                    prev = line
            if len(prev) > 0:
                print_sents(prev, "")
    bpein = "{}.{}".format(subset, lng)
    docin = "{}{}.{}.doc.in".format(subset, size_str, lng)
    print("Paste: %s + %s  ==>  %s" % (bpein, tgtfn, docin))
    paste_cmd = "paste -d \"\n\" {} {} > {}".format(bpein, tgtfn, docin)
    print("cmd: %s" % paste_cmd)
    os.system(paste_cmd)

