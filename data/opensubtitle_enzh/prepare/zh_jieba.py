# encoding=utf-8
import jieba
import argparse
import fileinput
parser = argparse.ArgumentParser('jieba split for a file or from stdin')
parser.add_argument('--srcfile', type=str, default='-')
args = parser.parse_args()
srcfile = args.srcfile
with fileinput.input(files=[srcfile], openhook=fileinput.hook_encoded("utf-8")) as h:
    for line in h:
        seg_list = jieba.cut(line.strip())
        print('{}'.format(' '.join(seg_list)))

