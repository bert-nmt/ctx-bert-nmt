import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('oup')
    parser.add_argument('src', choices=['en', 'zh_cn'])
    parser.add_argument('tgt', choices=['en', 'zh_cn'])
    return parser.parse_args()


def parse_doc(lines, token):
    docs = []
    idx = 0
    for line in lines:
        if line.strip() == token:
            docs.append(idx)
        else:
            idx += 1
    return docs


if __name__ == "__main__":
    args = parse_args()
    src = args.src
    tgt = args.tgt
    inp_path = args.inp
    oup_path = args.oup

    token = "startofdocumentplaceholder"

    x = "test"
    with open(os.path.join(inp_path, x + '.' + 'zh_cn.with_startofdoc'), encoding='utf-8') as f:
        lines = f.readlines()
    docs = parse_doc(lines, token)

    with open(os.path.join(inp_path, x + '.' + 'en.with_startofdoc'), encoding='utf-8') as f:
        lines = f.readlines()
    assert docs == parse_doc(lines, token)

    with open(os.path.join(oup_path, x + '.doc_index'), 'w', encoding='utf-8') as f:
        for idx in docs:
            f.write("{:d}\n".format(idx))
