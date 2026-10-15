import re
CANON = [(r"^(above|over)$","above"),(r"^(below|under|underneath|beneath)$","below"),
 (r"^(on|on top of|sitting on|standing on|lying on|placed on|resting on|mounted on)$","on"),
 (r"^in front of$","front"),(r"^behind$","behind"),(r"^(in|inside|within)$","inside"),
 (r"^(near|beside|next to|close to|adjacent to)$","near"),(r"^(attached to|part of)$","attached"),
 (r"^(approach|approaches|approaching|moving toward|moves toward|moving towards|moves towards)$","approach"),
 (r"^(moves away from|moving away from)$","away")]
def canon(p):
    p = p.strip().lower()
    for rx, n in CANON:
        if re.match(rx, p): return n
    return None
