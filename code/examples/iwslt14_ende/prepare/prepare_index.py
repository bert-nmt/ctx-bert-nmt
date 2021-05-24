import sys
import os


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
    _, inp_path, oup_path = sys.argv
    print("%s ==> %s" % (inp_path, oup_path))
    token = "startofdocumentplaceholder"

    for x in ["train", "valid", "test", ]:

        with open(os.path.join(inp_path, x + '.' + 'de.with_startofdoc'), encoding='utf-8') as f:
            lines = f.readlines()
        docs = parse_doc(lines, token)

        with open(os.path.join(inp_path, x + '.' + 'en.with_startofdoc'), encoding='utf-8') as f:
            lines = f.readlines()
        assert docs == parse_doc(lines, token)

        with open(os.path.join(oup_path, x + '.doc_index'), 'w', encoding='utf-8') as f:
            for idx in docs:
                f.write("{:d}\n".format(idx))
