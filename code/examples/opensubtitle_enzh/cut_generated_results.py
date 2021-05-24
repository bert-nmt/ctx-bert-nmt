import sys
import os
import numpy as np


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python xxx.py [inp] [oup]"
    _, inp, oup = sys.argv
    assert os.path.exists(inp), "{} doesn't exist".format(inp)

    with open(inp) as f:
        lines = f.readlines()
    lines = [x for x in lines if x.startswith('H-')]

    idxes = np.array([x.split()[0][2:] for x in lines], dtype=np.int)
    texts = [x.split('\t')[-1].strip() for x in lines]
    sort_idxes = np.argsort(idxes)

    with open(oup, 'w') as f:
        for i in sort_idxes:
            t = texts[i]
            f.write(t + '\n')

