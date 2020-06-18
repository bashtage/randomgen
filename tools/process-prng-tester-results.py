import json

import pandas as pd

with open("results.json") as res:
    results = json.load(res)

table = {}
for key in results:
    row = {}
    for size in results[key]:
        row[size] = "✔" if "FAIL" not in results[key][size] else "✖"
    table[key] = pd.Series(row)
results = pd.DataFrame(table).T
results = results.stack()

new_index = []
for key in results.index:
    base = key[0]
    jumped = False
    seed_seq = False
    if "-streams-" in base:
        jumped = "jumped" in base
        seed_seq = not jumped
        base = base.split("-")

        new_base = []
        for token in base:
            if token == "jumped" or token == "streams":
                break
            new_base.append(token)
        base = "-".join(new_base)
    new_base = base.split("-")
    if "cm" in new_base:
        for i in range(len(new_base)):
            new_base[i] = "cm-dxsm" if new_base[i] == "cm" else new_base[i]
        new_base.remove("dxsm")
    if "xsl" in new_base:
        for i in range(len(new_base)):
            new_base[i] = "xsl-rr" if new_base[i] == "xsl" else new_base[i]
        new_base.remove("rr")
    base = new_base[0]
    if len(new_base) > 1:
        base += "("
        for i in range(1, len(new_base), 2):
            base += new_base[i] + "=" + new_base[i + 1] + ", "
        base = base[:-2] + ")"

    size = key[1]
    if jumped:
        key = (base, "Jumped", key[0].split("-")[-1], size)
    elif seed_seq:
        key = (base, "SeedSeq", key[0].split("-")[-1], size)
    else:
        key = (base, "", "", size)

    new_index.append(key)

results.index = pd.MultiIndex.from_tuples(new_index)
final = results.unstack(level=[1, 2, 3]).fillna("--")


def find_max(s):
    if (s == "--").all():
        return "--"
    for v in s.index:
        if s.loc[v] != "✔":
            s.loc[v] = -1
        else:
            end = v[-2:]
            scale = 1 if end == "GB" else 1024
            s.loc[v] = int(v[:-2]) * scale
    return s.astype("i8").idxmax()


keys = [
    ("", ""),
    ("SeedSeq", "4"),
    ("SeedSeq", "8196"),
    ("Jumped", "4"),
    ("Jumped", "8196"),
]
new_table = {}
for bg in final.index:
    row = final.loc[bg]
    repl = {}
    for k in keys:
        repl[k] = find_max(row.loc[k])
    new_table[bg] = pd.Series(repl)
new_table = pd.DataFrame(new_table).T


columns = new_table.columns
header = ["Bit Generator", "", "4", "8196", "4", "8196"]
widths = list(map(lambda s: 2 + len(s), header))
widths[0] = max(widths[0], max(map(lambda s: 2 + len(s), new_table.index)))
widths = [max(w, 12) for w in widths]

first = "| Method"
first += " " * (widths[0] - len(first))
last_col = None
temp = ""
cum_width = 0
for c, w in zip(columns.droplevel(1), widths[1:]):
    if c != last_col:
        if last_col is not None:
            first += temp + " " * (cum_width - len(temp))
            cum_width = 0
        last_col = c
        temp = "| " + c
        cum_width += w + 1
    else:
        cum_width += w + 1
first += temp + " " * (cum_width - len(temp)) + "|"

second = "| Streams" + " " * (widths[0] - 9)
last_col = None
temp = ""
cum_width = 0
for c, w in zip(columns.droplevel(0), widths[1:]):
    if c != last_col:
        if last_col is not None:
            second += temp + " " * (cum_width - len(temp))
            cum_width = 0
        last_col = c
        temp = "| " + c
        cum_width += w + 1
    else:
        cum_width += w + 1
second += temp + " " * (cum_width - len(temp)) + "|"

rows = [first, second]
for row in new_table.T:
    s = list(new_table.T[row])
    parts = second.split("|")
    new_parts = []
    for i, (p, val) in enumerate(zip(list(parts[2:-1]), s)):
        val = val + " "
        val = " " * (len(p) - len(val)) + val
        assert len(val) == len(p)
        new_parts.append(val)
    bg = " " + row
    bg = bg + " " * (len(parts[1]) - len(bg))
    new_parts = ["", bg] + new_parts + [""]
    assert len(new_parts) == len(parts)
    assert tuple(map(len, new_parts)) == tuple(map(len, parts))
    rows.append("|".join(new_parts))

out = []
for i in range(len(rows)):
    before = rows[i]
    after = rows[i + 1] if i < (len(rows) - 1) else rows[i]
    if i == 0:
        temp = ""
        for c in before:
            temp += "+" if c == "|" else "-"
        out.append(temp)
    out.append(before)
    temp = ""
    for c1, c2 in zip(before, after):
        temp += "+" if c1 == "|" or c2 == "|" else "-"
    out.append(temp)

print("\n".join(out))
