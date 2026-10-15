"""Assemble a Kaggle (or local smoke) script: head with ENGINE_SRC replaced by engine/traser_fast.py, then extra parts.
Usage: python kaggle/build.py out.py head.py [part.py ...]"""
import sys
eng = open("/home/karthik/svg2-killtest/engine/traser_fast.py").read()
out, head, *parts = sys.argv[1:]
s = open(head).read().replace("ENGINE_SRC", eng)
for p in parts: s += "\n" + open(p).read()
open(out, "w").write(s)
