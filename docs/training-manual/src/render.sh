#!/bin/bash
# build docx -> add section bidi -> pdf -> page map -> (optional second pass) -> thumbnails
set -e
S=/tmp/claude-0/-home-user-gpt/d14ecac4-81bf-5b67-94fb-a4e393252daf/scratchpad
cd "$S"
export HOME="$S/lohome"

post() {
python3 - <<'PY'
import zipfile, shutil
src='out.docx'; tmp='out_fixed.docx'
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data=zin.read(item.filename)
    if item.filename=='word/document.xml':
        data=data.decode('utf8').replace('<w:docGrid','<w:bidi/><w:docGrid').encode('utf8')
    zout.writestr(item,data)
zout.close(); zin.close(); shutil.move(tmp,src)
PY
}

pagemap() {
python3 - <<'PY'
import pymupdf, json, re
d=pymupdf.open('out.pdf')
toc=json.load(open('toc.json'))
m={}
pages=[d[i].get_text() for i in range(len(d))]
# chapters: banner pages contain 'פרק NN'
for i,t in enumerate(pages):
    for mm in re.finditer(r'פרק\s*(\d\d)', t):
        k='c'+str(int(mm.group(1)))
        if k not in m: m[k]=i+1-2
# H2 headings: exact line match, searched in order after the previous hit
def toks(x):
    x=re.sub(r'([\u0590-\u05FF])([A-Za-z0-9])', r'\1 \2', x); x=re.sub(r'([A-Za-z0-9])([\u0590-\u05FF])', r'\1 \2', x)
    return sorted(re.findall(r'[\w]+', x))
BODY_START=2
pos=BODY_START
for e in toc:
    if e['level']!=2: continue
    title=e['title'].strip(); tt=toks(title)
    found=None
    for i in range(pos, len(pages)):
        L=[x for x in pages[i].split('\n') if x.strip()]
        for a in range(len(L)):
            for b in range(a+1, min(a+4, len(L)+1)):
                if toks(' '.join(L[a:b]))==tt:
                    found=i; break
            if found is not None: break
        if found is not None: break
    if found is None:
        st=set(tt)
        for i in range(pos, len(pages)):
            L=[x for x in pages[i].split('\n') if x.strip()]
            for a in range(len(L)):
                for b in range(a+1, min(a+4, len(L)+1)):
                    lt=toks(' '.join(L[a:b]))
                    if st.issubset(set(lt)) and len(lt)<=len(tt)+2:
                        found=i; break
                if found is not None: break
            if found is not None: break
    if found is None:
        st=set(tt)
        for i in range(pos, len(pages)):
            L=[x for x in pages[i].split('\n') if x.strip()]
            for a in range(len(L)):
                if st.issubset(set(toks(' '.join(L[a:a+3])))):
                    found=i; break
            if found is not None: break
    if found is not None:
        m['h:'+title]=found+1-2; pos=found
    else:
        print('NOT FOUND:', title)
json.dump(m, open('pagemap.json','w'), ensure_ascii=False)
print('pagemap entries', len(m), 'pages', len(d))
PY
}

convert() {
  rm -f out.pdf
  timeout 400 soffice --headless --convert-to pdf "$S/out.docx" --outdir "$S" >/dev/null 2>&1
}

node build.js && post && convert && pagemap
node build.js && post && convert && pagemap
python3 - <<'PY'
import pymupdf
from PIL import Image
d=pymupdf.open('out.pdf'); n=len(d)
ims=[]
for i in range(n):
    ims.append(Image.frombytes('RGB', [d[i].get_pixmap(dpi=26).width, d[i].get_pixmap(dpi=26).height], d[i].get_pixmap(dpi=26).samples))
w,h=ims[0].size; cols=8; rows=(n+cols-1)//cols
s=Image.new('RGB',(w*cols+4*(cols+1),h*rows+4*(rows+1)),(200,200,200))
for k,im in enumerate(ims): s.paste(im,(4+(k%cols)*(w+4),4+(k//cols)*(h+4)))
s.save('thumbs.png'); print('thumbs', s.size, 'pages', n)
PY
