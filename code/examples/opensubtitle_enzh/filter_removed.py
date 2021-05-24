import sys


_, inp, ref = sys.argv
with open(inp) as f: inp = f.readlines()
with open(ref) as f: ref = f.readlines()
assert len(inp) == len(ref)

for i, r in zip(inp, ref):
    if r.strip() != "_removed_":
        print(i.strip())
