"""out.pdf -> final.pdf : metadata, RTL viewer prefs, full bookmark outline, clickable TOC."""
import pymupdf, json, re, sys

SRC, DST = 'out.pdf', 'final.pdf'
d = pymupdf.open(SRC)
n = len(d)
BODY = 2  # cover + toc pages before body page 1

def toks(x):
    x = re.sub(r'([֐-׿])([A-Za-z0-9])', r'\1 \2', x)
    x = re.sub(r'([A-Za-z0-9])([֐-׿])', r'\1 \2', x)
    return sorted(re.findall(r'\w+', x))

pages = [d[i].get_text() for i in range(n)]
page_lines = [[l for l in p.split('\n') if l.strip()] for p in pages]

def find(title, start):
    tt = toks(title); st = set(tt)
    for i in range(start, n):
        L = page_lines[i]
        for a in range(len(L)):
            for b in range(a + 1, min(a + 4, len(L) + 1)):
                if toks(' '.join(L[a:b])) == tt:
                    return i
    for i in range(start, n):
        L = page_lines[i]
        for a in range(len(L)):
            for b in range(a + 1, min(a + 4, len(L) + 1)):
                lt = toks(' '.join(L[a:b]))
                if st.issubset(set(lt)) and len(lt) <= len(tt) + 2:
                    return i
    return None

# ---- collect structure from the content files (chapter > H2 > H3 / drugs)
struct = []  # (level, title)
for f in ['part1.txt', 'part2.txt', 'part3.txt', 'part4.txt', 'part5.txt']:
    for ln in open('content/' + f, encoding='utf8'):
        t = ln.strip()
        m = re.match(r'^\[chapter\]\s*(.+)$', t)
        if m: struct.append((1, m.group(1).strip())); continue
        m = re.match(r'^##\s+(.+)$', t)
        if m: struct.append((2, m.group(1).strip())); continue
        m = re.match(r'^###\s+(.+)$', t)
        if m: struct.append((3, m.group(1).strip())); continue
        m = re.match(r'^\[drug\]\s*(.+)$', t)
        if m:
            p = [s.strip() for s in m.group(1).split('|')]
            struct.append((3, f"{p[0]} ({p[2]})", p[2])); continue
        m = re.match(r'^\[final\]\s*(.+)$', t)
        if m: struct.append((1, m.group(1).strip())); continue

# chapters by banner "פרק NN"
chap_pages = {}
for i, t in enumerate(pages):
    for mm in re.finditer(r'פרק\s*(\d\d)', t):
        k = int(mm.group(1))
        if k not in chap_pages: chap_pages[k] = i

outline = [[1, 'שער', 1], [1, 'תוכן העניינים', 2]]
pos = BODY
chap_no = 0
missing = []
for item in struct:
    level, title = item[0], item[1]
    if level == 1:
        if title.startswith('זכרו'):
            pg = find(title, pos)
        else:
            chap_no += 1
            pg = chap_pages.get(chap_no)
    elif len(item) == 3:  # drug: search by latin name
        latin = item[2]
        pg = next((i for i in range(pos, n) if any(l.strip() == latin for l in page_lines[i]) and 'דרכי מתן' in pages[i]), None)
    else:
        pg = find(title, pos)
    if pg is None:
        missing.append(title); continue
    pos = max(pos, pg)
    clean = re.sub(r'\*\*', '', title).rstrip(':')
    outline.append([level, clean, pg + 1])

# pymupdf requires each level to be at most parent+1: fix jumps
fixed = []
prev = 0
for lvl, t, p in outline:
    if lvl > prev + 1: lvl = prev + 1
    fixed.append([lvl, t, p]); prev = lvl
d.set_toc(fixed)

# ---- clickable table of contents (page index 1)
toc_page = d[1]
links = 0
for lvl, t, p in fixed:
    if lvl > 2 or p <= 2: continue
    hits = toc_page.search_for(t[:40]) or toc_page.search_for(re.sub(r'\(.*?\)', '', t).strip()[:30])
    if not hits: continue
    r = hits[0]
    rect = pymupdf.Rect(toc_page.rect.x0 + 40, r.y0 - 2, toc_page.rect.x1 - 40, r.y1 + 2)
    toc_page.insert_link({'kind': pymupdf.LINK_GOTO, 'from': rect, 'page': p - 1, 'to': pymupdf.Point(0, 0)})
    links += 1

# ---- metadata + viewer prefs
d.set_metadata({'title': 'תוכנית הכשרה לטכנאי הדמיה', 'author': 'רשת "חוות דעת" – מחלקת הדמיה',
                'subject': 'מדריך הכשרה למחלקת ההדמיה', 'creator': '', 'producer': ''})
cat = d.pdf_catalog()
d.xref_set_key(cat, 'ViewerPreferences', '<< /Direction /R2L /DisplayDocTitle true >>')
d.xref_set_key(cat, 'PageMode', '/UseOutlines')
d.save(DST, garbage=3, deflate=True)
print(f'outline entries: {len(fixed)}, toc links: {links}, missing: {missing}')
