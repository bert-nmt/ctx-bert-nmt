import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('tgt')
    parser.add_argument('ids')
    parser.add_argument('--doc-dir', default=None)
    args = parser.parse_args()
    if args.doc_dir is None:
        assert os.path.dirname(args.src) == os.path.dirname(args.tgt)
        args.doc_dir = os.path.join(os.path.dirname(args.src), "docs")
    if not os.path.exists(args.doc_dir):
        os.makedirs(args.doc_dir)
    return args


def write_doc(doc, idx, args):
    fn_src = os.path.join(args.doc_dir, os.path.basename(args.src) + ".doc_{:d}".format(i))
    fn_tgt = os.path.join(args.doc_dir, os.path.basename(args.tgt) + ".doc_{:d}".format(i))
    with open(fn_src, 'w', encoding='utf-8') as fsrc:
        with open(fn_tgt, 'w', encoding='utf-8') as ftgt:
            for s, t in doc:
                fsrc.write(s.strip() + '\n')
                ftgt.write(t.strip() + '\n')


if __name__ == "__main__":
    args = parse_args()
    print(args)

    with open(args.src, encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(args.tgt, encoding='utf-8') as f:
        tgt_lines = f.readlines()
    with open(args.ids, encoding='utf-8') as f:
        ids = f.readlines()
    assert len(src_lines) == len(tgt_lines) == len(ids)

    doc_lines = []
    prev_doc = ""
    for i, line in enumerate(ids):
        doc = ' '.join(line.split('\t')[:2])
        if doc != prev_doc:
            doc_lines.append([])
        doc_lines[-1].append((src_lines[i], tgt_lines[i]))
        prev_doc = doc
    print("Total: %d documents" % len(doc_lines))

    for i, d in tqdm(list(enumerate(doc_lines))):
        write_doc(d, i, args)

