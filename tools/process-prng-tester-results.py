import json

import pandas as pd

with open("results.json") as res:
    results = json.load(res)


def parse_key(key):
    if "-" not in key:
        return key, "", ""
    if "-jumped-streams-" in key:
        base, n = key.split("-jumped-streams-")
        method = "Jumped"
    elif "-streams-" in key:
        base, n = key.split("-streams-")
        method = "Seed Sequence"
    else:
        base = key
        method = n = ""
    base = base.replace("xsl-rr", "xsl_rr").replace("dxsm-128", "dxsm_128")
    parts = base.split("-")
    base = parts[0]
    for i in range(1, len(parts), 2):
        if i == 1:
            base += "("
        base += f"{parts[i]}={parts[i+1]}, "
    if "(" in base:
        base = f"{base[:-2]})"
    return base, method, n


def to_bytes(s):
    if "GB" in s:
        return int(s[:-2].strip()) * 2 ** 30
    elif "TB" in s:
        return int(s[:-2].strip()) * 2 ** 40


def from_bytes(b):
    b = b >> 30
    if b >= 2 ** 10:
        return f"{b>>10}TB"
    else:
        return f"{b}GB"


series_data = {}
for key in results:
    parsed_key = parse_key(key)
    result = results[key]
    result = sorted(
        [(to_bytes(key), result[key]) for key in result],
        key=lambda v: v[0],
        reverse=True,
    )
    if "FAIL" not in result[0][1]:
        series_data[parsed_key] = from_bytes(result[0][0])
    else:
        lines = result[0][1].split("\n")
        lines = lines[::-1]
        max_pass = None
        for i in range(len(lines) - 1):
            if "no anomalies" in lines[i]:
                max_line = lines[i + 1]
                max_pass = "".join(max_line.split(" ")[1:3])
                max_pass = (
                    max_pass.replace("giga", "G")
                    .replace("bytes", "B")
                    .replace("tera", "T")
                )
                # ❌
                if max_pass == "256GB":
                    max_pass = "FAIL¹"
                else:
                    raise NotImplementedError("need to add footnote")

                break
        series_data[parsed_key] = max_pass
series = pd.Series(series_data)
df = series.unstack([1, 2]).fillna("--")
df.index = [val.replace("_", "-") for val in df.index]

replacements = {
    "DSFMT": "DSFMT⁴",
    "MT19937": "MT19937⁴,⁵",
    "Philox": "Philox⁵",
    "SFC64(k=1)": "SFC64⁵",
    "SFC64(k=weyl)": "SFC64(k=Weyl)³",
    "SFMT": "SFMT⁴",
    "PCG64(variant=xsl-rr)": "PCG64⁵",
    "PCG64(variant=dxsm)": "PCG64DXSM²",
    "JSF(seed-size=1)": "JSF",
    "JSF(seed-size=3)": "JSF(seed_size=3)",
}
df.index = [replacements.get(key, key) for key in df.index]

keys = [
    ("", ""),
    ("Seed Sequence", "4"),
    ("Seed Sequence", "8196"),
    ("Jumped", "4"),
    ("Jumped", "8196"),
]
new_table = df[keys]


columns = new_table.columns
header = ["Bit Generator", "", "4", "8196", "4", "8196"]
widths = list(map(lambda s: 2 + len(s), header))
widths[0] = max(widths[0], max(map(lambda s: 2 + len(s), new_table.index)) + 1)
widths = [max(w, 11) for w in widths]

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

# Fix header setp
out[4] = out[4].replace("-", "=")

print("\n".join(out))
