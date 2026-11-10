"""Writes <job>.bbl in the Harvard reference style of the Journal of Quantitative Analysis in Sports
(Reference Style Sheet: 'Surname, I.I. and Surname, I.I. (Year). Title. Journal Vol: pages, https://doi.org/...';
no issue numbers; unspaced initials; more than ten authors -> first ten + et al.). Replaces BibTeX.
Usage: python code/make_bbl.py paper/refs.bib paper/main_dg.aux paper/main_dg.bbl"""
import re, sys, unicodedata

def parse_bib(text):
    out = {}; i = 0
    while True:
        m = re.compile(r'@(\w+)\s*\{\s*([^,\s]+)\s*,').search(text, i)
        if not m: break
        j = m.end(); depth = 1; k = j
        while depth:
            c = text[k]; depth += (c == '{') - (c == '}'); k += 1
        body = text[j:k - 1]; fields = {}; p = 0
        while True:
            fm = re.compile(r'\s*(\w+)\s*=\s*').match(body, p)
            if not fm: break
            q = fm.end()
            if body[q] == '{':
                d = 1; r = q + 1
                while d:
                    d += (body[r] == '{') - (body[r] == '}'); r += 1
                val = body[q + 1:r - 1]
            else:
                r = q
                while r < len(body) and body[r] not in ',': r += 1
                val = body[q:r].strip().strip('"')
            fields[fm.group(1).lower()] = re.sub(r'\s+', ' ', val).strip(); p = r
            while p < len(body) and body[p] in ', \n\t': p += 1
        out[m.group(2)] = (m.group(1).lower(), fields); i = k
    return out

def split_names(s):
    parts, depth, cur = [], 0, ''
    toks = re.split(r'(\s+and\s+)', s)
    buf = ''
    for t in toks:
        if re.fullmatch(r'\s+and\s+', t) and buf.count('{') == buf.count('}'):
            parts.append(buf); buf = ''
        else: buf += t
    parts.append(buf); return [p.strip() for p in parts if p.strip()]

def initials(first):
    out = []
    for w in first.split():
        segs = w.split('-'); ini = []
        for sgm in segs:
            if not sgm: continue
            if sgm.startswith('{\\') or sgm.startswith('\\'):
                m = re.match(r'(\{\\[^{}]*(?:\{[^{}]*\})?\}|\\[^\w\s]\{?\w\}?|\\\w+\{\w\})', sgm)
                ini.append((m.group(1) if m else sgm[:3]) + '.')
            elif sgm.startswith('{'):
                ini.append(sgm[1] + '.')
            else:
                ini.append(sgm[0] + '.')
        out.append('-'.join(ini))
    return ''.join(out)

def person(n):
    if n.startswith('{') and n.endswith('}') and n.count('{') == n.count('}') and ',' not in n[1:-1]:
        return n[1:-1], None
    if ',' in n:
        last, first = [x.strip() for x in n.split(',', 1)]
    else:
        ws = n.split(); last, first = ws[-1], ' '.join(ws[:-1])
    return last, initials(first) if first else ''

def fmt_authors(a):
    names = [person(x) for x in split_names(a)]
    full = [f'{l}, {i}' if i else l for l, i in names]
    if len(full) > 10: return ', '.join(full[:10]) + ', et al.', names
    if len(full) == 1: return full[0], names
    if len(full) == 2: return f'{full[0]} and {full[1]}', names
    return ', '.join(full[:-1]) + ', and ' + full[-1], names

def plain(s):
    s = re.sub(r'(?<![A-Za-z\\])\{([^{}\\]*)\}', r'\1', s)
    s = re.sub(r'(?<![A-Za-z\\])\{([^{}\\]*)\}', r'\1', s)
    return s

def sortkey(s):
    s = re.sub(r'\\.', '', s); s = s.replace('{', '').replace('}', '')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()

def endp(t): return t if t.rstrip().endswith(('?', '!', '.')) else t + '.'

def doi(f): return f", \\url{{https://doi.org/{f['doi']}}}" if 'doi' in f else ''

def entry(typ, f):
    au, names = fmt_authors(f['author']); t = plain(f['title']); y = f['year']
    head = f'{au} ({y}). '
    if typ == 'article':
        vol = f.get('volume', ''); pg = f.get('pages', '')
        src = plain(f['journal']) + (f' {vol}' if vol else '') + (f': {pg}' if pg else '')
        return head + endp(t) + ' ' + src + doi(f) + '.', names
    if typ == 'book':
        return head + endp(t) + f" {f.get('publisher', '')}, {f.get('address', '')}.", names
    if typ in ('inproceedings', 'incollection'):
        ed = ''
        if 'editor' in f:
            e, en = fmt_authors(f['editor']); ed = e + (' (Eds.), ' if len(en) > 1 else ' (Ed.), ')
        bits = [plain(f['booktitle'])]
        if 'series' in f: bits.append(plain(f['series']) + (f" {f['volume']}" if 'volume' in f else ''))
        pub = ', '.join(x for x in [f.get('publisher'), f.get('address')] if x)
        s = head + endp(t) + ' In: ' + ed + ', '.join(bits)
        if pub: s += '. ' + pub
        if 'pages' in f: s += f", pp. {f['pages']}"
        return s + doi(f) + '.', names
    # misc: preprints and web resources
    hp = f.get('howpublished', '')
    m = re.match(r'\\url\{(.*)\}', hp)
    if m:
        acc = re.sub(r'^Accessed\s*', '', f.get('note', ''))
        return head + t + f', Available at: \\url{{{m.group(1)}}}' + (f' (Accessed {acc}).' if acc else '.'), names
    return head + endp(t) + ' ' + hp + doi(f) + '.', names

def label(names, y):
    ls = [l for l, _ in names]
    if len(ls) == 1: short = ls[0]
    elif len(ls) == 2: short = f'{ls[0]} and {ls[1]}'
    else: short = f'{ls[0]} et~al.'
    return short, y

bib = parse_bib(open(sys.argv[1]).read())
aux = open(sys.argv[2]).read(); cited = []
for m in re.findall(r'\\citation\{([^}]*)\}', aux):
    for k in m.split(','):
        if k not in cited: cited.append(k)
items = []
for k in cited:
    typ, f = bib[k]; txt, names = entry(typ, f); short, y = label(names, f['year'])
    items.append(dict(key=k, txt=txt, short=short, year=y, sk=(sortkey(' '.join(l for l, _ in names)), y)))
items.sort(key=lambda d: d['sk'])
seen = {}
for d in items:
    lab = (d['short'], d['year']); seen.setdefault(lab, []).append(d)
for lab, ds in seen.items():
    if len(ds) > 1:
        for n, d in enumerate(ds): d['year'] = d['year'] + 'abcdefgh'[n]; d['txt'] = d['txt'].replace(f"({lab[1]})", f"({d['year']})", 1)
with open(sys.argv[3], 'w') as o:
    o.write(f'\\begin{{thebibliography}}{{{len(items)}}}\n\n')
    for d in items: o.write(f"\\bibitem[{{{d['short']}({d['year']})}}]{{{d['key']}}}\n{d['txt']}\n\n")
    o.write('\\end{thebibliography}\n')
print(len(items), 'references written')
