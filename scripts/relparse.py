import json, re
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]?)')
NUM2 = re.compile(r'\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]')
def parse_rels(text):
    try:
        r = json.loads(text)["relationships"]
        if isinstance(r, list): return r
    except Exception: pass
    out = []
    for a, p, b, s in REL_RX.findall(text):
        spans = [[float(x), float(y)] for x, y in NUM2.findall(s)]
        out.append([int(a), p, int(b), spans])
    return out
