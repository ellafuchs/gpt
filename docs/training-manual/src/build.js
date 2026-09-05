const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, HeadingLevel, PageBreak,
  Header, Footer, PageNumber, LevelFormat, ExternalHyperlink, VerticalAlign,
  convertMillimetersToTwip, TableLayoutType, PageOrientation, TabStopType,
} = d;

const DIR = __dirname;
const FONT = 'Arial';

// ---------- palette ----------
const C = {
  ink: '1A2B30',
  body: '23383D',
  primary: '14535F',
  primaryLight: '2E7F8C',
  soft: 'E7F0F1',
  softer: 'F3F8F8',
  rule: 'C9DBDD',
  grayText: '6B7F83',
  dangerLine: 'B3261E', dangerBg: 'FBEDEB', dangerText: '8C1D18',
  warnLine: 'C77700', warnBg: 'FFF6E6', warnText: '7A4B00',
  noteLine: '2E7F8C', noteBg: 'EAF4F6', noteText: '14535F',
  tipLine: '3C7D4F', tipBg: 'EDF6EF', tipText: '2A5C39',
  prosBg: 'EFF7F0', prosLine: '3C7D4F',
  consBg: 'FBF0EE', consLine: 'B3261E',
  white: 'FFFFFF',
};

const A4_W = 11906, A4_H = 16838;
const MARGIN = convertMillimetersToTwip(20);
const CONTENT_W = A4_W - MARGIN * 2;

// ---------- helpers ----------
const noBorder = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder,
  insideHorizontal: noBorder, insideVertical: noBorder };

function runs(text, opts = {}) {
  const base = { font: FONT, rightToLeft: true, size: opts.size || 22, color: opts.color || C.body };
  const out = [];
  const parts = String(text).split('**');
  parts.forEach((p, i) => {
    if (p === '') return;
    out.push(new TextRun({ ...base, text: p, bold: opts.bold || (i % 2 === 1) }));
  });
  if (out.length === 0) out.push(new TextRun({ ...base, text: '' }));
  return out;
}

function P(text, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { before: opts.before === undefined ? 0 : opts.before,
               after: opts.after === undefined ? 120 : opts.after,
               line: opts.line || 300 },
    indent: opts.indent,
    keepNext: opts.keepNext,
    border: opts.border,
    shading: opts.shading,
    children: opts.children || runs(text, opts),
  });
}

function img(file, wPt, hPt, align) {
  return new Paragraph({
    bidirectional: true,
    alignment: align || AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new ImageRun({
      type: 'png',
      data: fs.readFileSync(path.join(DIR, file)),
      transformation: { width: wPt, height: hPt },
    })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(path.join(DIR, file));
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function cell(children, opts = {}) {
  return new TableCell({
    children,
    width: { size: opts.width || 100, type: WidthType.PERCENTAGE },
    shading: opts.fill ? { type: ShadingType.CLEAR, fill: opts.fill, color: 'auto' } : undefined,
    margins: { top: opts.mt === undefined ? 90 : opts.mt, bottom: opts.mb === undefined ? 90 : opts.mb,
               left: opts.ml === undefined ? 140 : opts.ml, right: opts.mr === undefined ? 140 : opts.mr },
    borders: opts.borders,
    columnSpan: opts.span,
    verticalAlign: VerticalAlign.CENTER,
  });
}

function oneCellBox(children, { bg, line }) {
  const borders = {
    top: { style: BorderStyle.SINGLE, size: 2, color: bg },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: bg },
    left: { style: BorderStyle.SINGLE, size: 24, color: line },
    right: { style: BorderStyle.SINGLE, size: 2, color: bg },
  };
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [CONTENT_W],
    layout: TableLayoutType.FIXED,
    borders: noBorders,
    rows: [new TableRow({
      cantSplit: true,
      children: [new TableCell({
        children,
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: bg, color: 'auto' },
        borders,
        margins: { top: 150, bottom: 150, left: 220, right: 220 },
      })],
    })],
  });
}

function spacer(h) { return new Paragraph({ spacing: { before: 0, after: h || 120 }, children: [] }); }

// ---------- numbering ----------
const numbering = { config: [] };
for (let i = 0; i < 200; i++) {
  numbering.config.push({
    reference: `num-${i}`,
    levels: [{
      level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.START,
      style: { paragraph: { indent: { start: 460, hanging: 300 } },
               run: { font: FONT, color: C.primary, bold: true } },
    }],
  });
}
numbering.config.push({
  reference: 'bul',
  levels: [
    { level: 0, format: LevelFormat.BULLET, text: '●', alignment: AlignmentType.START,
      style: { paragraph: { indent: { start: 400, hanging: 260 } },
               run: { font: FONT, color: C.primaryLight, size: 16 } } },
    { level: 1, format: LevelFormat.BULLET, text: '○', alignment: AlignmentType.START,
      style: { paragraph: { indent: { start: 780, hanging: 260 } },
               run: { font: FONT, color: C.primaryLight, size: 16 } } },
  ],
});

// ---------- parse ----------
const files = ['part1.txt', 'part2.txt', 'part3.txt', 'part4.txt', 'part5.txt'];
let lines = [];
for (const f of files) {
  lines = lines.concat(fs.readFileSync(path.join(DIR, 'content', f), 'utf8').split('\n'));
}

const body = [];
let numIdx = 0;
let firstDrug = true;
let inNum = false;
const toc = [];
let chapterNo = 0;

function headingPara(text, level) {
  if (level === 2) {
    return new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.START,
      spacing: { before: 320, after: 140, line: 280 },
      keepNext: true,
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.rule, space: 6 } },
      children: [new TextRun({ font: FONT, rightToLeft: true, text, bold: true, size: 28, color: C.primary })],
    });
  }
  if (level === 3) {
    return new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.START,
      spacing: { before: 240, after: 100, line: 280 },
      keepNext: true,
      children: [new TextRun({ font: FONT, rightToLeft: true, text, bold: true, size: 24, color: C.primaryLight })],
    });
  }
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.START,
    spacing: { before: 200, after: 80, line: 280 },
    keepNext: true,
    children: [new TextRun({ font: FONT, rightToLeft: true, text, bold: true, size: 22, color: C.ink })],
  });
}

function chapterBlock(title) {
  chapterNo += 1;
  toc.push({ title, no: chapterNo });
  const n = String(chapterNo).padStart(2, '0');
  const pre = chapterNo === 1 ? [] : [new Paragraph({ children: [new PageBreak()] })];
  return [
    ...pre,
    new Table({
      visuallyRightToLeft: true,
      width: { size: 100, type: WidthType.PERCENTAGE },
      columnWidths: [CONTENT_W],
      layout: TableLayoutType.FIXED,
      borders: noBorders,
      rows: [new TableRow({
        cantSplit: true,
        children: [new TableCell({
          width: { size: CONTENT_W, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: C.primary, color: 'auto' },
          borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
          margins: { top: 260, bottom: 260, left: 280, right: 280 },
          children: [
            new Paragraph({
              bidirectional: true, alignment: AlignmentType.START, spacing: { after: 40 },
              children: [new TextRun({ font: FONT, rightToLeft: true, text: `פרק ${n}`, bold: true, size: 18, color: 'A8D3DA' })],
            }),
            new Paragraph({
              bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
              children: [new TextRun({ font: FONT, rightToLeft: true, text: title, bold: true, size: 40, color: C.white })],
            }),
          ],
        })],
      })],
    }),
    spacer(200),
  ];
}

function calloutBlock(kind, title, text) {
  const map = {
    danger: { bg: C.dangerBg, line: C.dangerLine, tc: C.dangerText },
    warn: { bg: C.warnBg, line: C.warnLine, tc: C.warnText },
    note: { bg: C.noteBg, line: C.noteLine, tc: C.noteText },
    tip: { bg: C.tipBg, line: C.tipLine, tc: C.tipText },
  }[kind];
  const kids = [];
  if (title) {
    kids.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.START, spacing: { after: text ? 60 : 0, line: 300 },
      children: [new TextRun({ font: FONT, rightToLeft: true, text: title, bold: true, size: 23, color: map.tc })],
    }));
  }
  if (text) {
    kids.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 300 },
      children: runs(text, { size: 22, color: map.tc }),
    }));
  }
  return oneCellBox(kids, map);
}

function drugBlock(name, short, eng, routes) {
  const label = short ? `${name} (${short})` : name;
  const kids = [
    new Paragraph({
      bidirectional: true, alignment: AlignmentType.START, spacing: { after: 40 }, keepNext: true,
      children: [
        new TextRun({ font: FONT, rightToLeft: true, text: label, bold: true, size: 30, color: C.white }),
        new TextRun({ font: FONT, rightToLeft: true, text: `   ${eng}`, bold: false, size: 22, color: 'B8DDE3' }),
      ],
    }),
    new Paragraph({
      bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 }, keepNext: true,
      children: [new TextRun({ font: FONT, rightToLeft: true, text: `דרכי מתן:  ${routes}`, size: 20, color: 'D6ECEF' })],
    }),
  ];
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [CONTENT_W],
    layout: TableLayoutType.FIXED,
    borders: noBorders,
    rows: [new TableRow({
      cantSplit: true,
      children: [new TableCell({
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: C.primary, color: 'auto' },
        borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
        margins: { top: 180, bottom: 180, left: 240, right: 240 },
        children: kids,
      })],
    })],
  });
}

function prosConsBlock(pros, cons) {
  const half = Math.floor(CONTENT_W / 2);
  function col(titleText, txt, bg, line) {
    return new TableCell({
      width: { size: half, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: bg, color: 'auto' },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 12, color: line },
        bottom: { style: BorderStyle.SINGLE, size: 2, color: bg },
        left: { style: BorderStyle.SINGLE, size: 2, color: bg },
        right: { style: BorderStyle.SINGLE, size: 2, color: bg },
      },
      margins: { top: 130, bottom: 130, left: 180, right: 180 },
      children: [
        new Paragraph({
          bidirectional: true, alignment: AlignmentType.START, spacing: { after: 50 },
          children: [new TextRun({ font: FONT, rightToLeft: true, text: titleText, bold: true, size: 21, color: line })],
        }),
        new Paragraph({
          bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0, line: 290 },
          children: runs(txt, { size: 20 }),
        }),
      ],
    });
  }
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [half, half],
    layout: TableLayoutType.FIXED,
    borders: noBorders,
    rows: [new TableRow({ cantSplit: true, children: [
      col('יתרונות', pros, C.prosBg, C.prosLine),
      col('חסרונות', cons, C.consBg, C.consLine),
    ] })],
  });
}

function needlesBlock(items) {
  const swatch = { 'כתומה': 'E88B2E', 'כחולה': '3B7FC4', 'ירוקה': '3FA75C', 'וורודה': 'D4649B' };
  const w = Math.floor(CONTENT_W / items.length);
  const cells = items.map((it) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: swatch[it] || 'CCCCCC', color: 'auto' },
    borders: { top: noBorder, bottom: noBorder,
               left: { style: BorderStyle.SINGLE, size: 12, color: 'FFFFFF' },
               right: { style: BorderStyle.SINGLE, size: 12, color: 'FFFFFF' } },
    margins: { top: 150, bottom: 150, left: 80, right: 80 },
    children: [new Paragraph({
      bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ font: FONT, rightToLeft: true, text: it, bold: true, size: 21, color: 'FFFFFF' })],
    })],
  }));
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: items.map(() => w),
    layout: TableLayoutType.FIXED,
    borders: noBorders,
    rows: [new TableRow({ cantSplit: true, children: cells })],
  });
}

function tableBlock(widths, rows, colors) {
  const cols = widths.map((p) => Math.floor(CONTENT_W * p / 100));
  const trs = rows.map((cells, ri) => {
    const isHead = ri === 0;
    return new TableRow({
      tableHeader: isHead,
      cantSplit: true,
      children: cells.map((txt, ci) => {
        let fill = isHead ? C.primary : (ri % 2 === 0 ? C.softer : C.white);
        let color = isHead ? C.white : C.body;
        let bold = isHead;
        if (!isHead && colors && ci === 0 && colors[ri - 1]) { fill = colors[ri - 1]; color = C.white; bold = true; }
        return new TableCell({
          width: { size: cols[ci], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill, color: 'auto' },
          borders: {
            top: { style: BorderStyle.SINGLE, size: 2, color: C.rule },
            bottom: { style: BorderStyle.SINGLE, size: 2, color: C.rule },
            left: { style: BorderStyle.SINGLE, size: 2, color: C.rule },
            right: { style: BorderStyle.SINGLE, size: 2, color: C.rule },
          },
          margins: { top: 110, bottom: 110, left: 150, right: 150 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({
            bidirectional: true,
            alignment: ci === 0 ? AlignmentType.START : AlignmentType.START,
            spacing: { after: 0, line: 280 },
            children: runs(txt, { size: 20, color, bold }),
          })],
        });
      }),
    });
  });
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: cols,
    layout: TableLayoutType.FIXED,
    borders: noBorders,
    rows: trs,
  });
}

// ---------- main loop ----------
let i = 0;
let lastCallout = null;
while (i < lines.length) {
  const raw = lines[i];
  const line = raw.replace(/\s+$/, '');
  const t = line.trim();
  i++;

  if (t === '') { inNum = false; lastCallout = null; continue; }
  if (t === '[steps]' || t === '[/steps]') continue;

  let m;
  if ((m = t.match(/^\[chapter\]\s*(.+)$/))) {
    body.push(...chapterBlock(m[1].trim())); inNum = false; continue;
  }
  if ((m = t.match(/^\[final\]\s*(.+)$/))) {
    body.push(new Paragraph({ children: [new PageBreak()] }));
    body.push(spacer(600));
    body.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 260 },
      children: [new TextRun({ font: FONT, rightToLeft: true, text: m[1].trim(), bold: true, size: 48, color: C.dangerLine })],
    }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[drug\]\s*(.*)$/))) {
    const parts = m[1].split('|').map((s) => s.trim());
    if (!firstDrug) body.push(new Paragraph({ children: [new PageBreak()] }));
    firstDrug = false;
    body.push(spacer(60));
    body.push(drugBlock(parts[0], parts[1], parts[2], parts[3]));
    body.push(spacer(80));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[(danger|warn|note|tip)\]\s*(.*)$/))) {
    const kind = m[1];
    const rest = m[2];
    let title = null, text = rest;
    if (rest.includes('|')) {
      const idx = rest.indexOf('|');
      title = rest.slice(0, idx).trim();
      text = rest.slice(idx + 1).trim();
    }
    body.push(spacer(100));
    body.push(calloutBlock(kind, title, text));
    body.push(spacer(140));
    lastCallout = kind;
    inNum = false; continue;
  }
  if (t.startsWith('> ')) {
    const kindMap = { danger: [C.dangerBg, C.dangerLine, C.dangerText], warn: [C.warnBg, C.warnLine, C.warnText],
      note: [C.noteBg, C.noteLine, C.noteText], tip: [C.tipBg, C.tipLine, C.tipText] };
    const k = kindMap[lastCallout || 'note'];
    body.push(oneCellBox([new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 300 },
      children: runs(t.slice(2), { size: 22, color: k[2] }),
    })], { bg: k[0], line: k[1] }));
    body.push(spacer(140));
    continue;
  }
  if ((m = t.match(/^\[img\]\s*(.*)$/))) {
    const parts = m[1].split('|').map((s) => s.trim());
    const file = parts[0], caption = parts[1], url = parts[2];
    if (caption) body.push(P(caption, { alignment: AlignmentType.START, after: 60 }));
    const sz = pngSize(file);
    const h = 92, w = Math.round(92 * sz.w / sz.h);
    body.push(img(file, w, h));
    if (url) {
      body.push(new Paragraph({
        bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new ExternalHyperlink({
          link: url,
          children: [new TextRun({ font: FONT, text: 'לחצו כאן לצפייה בסרטון', style: 'Hyperlink', size: 19, rightToLeft: true })],
        })],
      }));
    }
    inNum = false; continue;
  }
  if ((m = t.match(/^\[formula\]\s*(.+)$/))) {
    const key = m[1].trim();
    if (key === 'f1') {
      body.push(oneCellBox([new Paragraph({
        bidirectional: false, alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [new TextRun({ font: FONT, text: 'משקל החיה (kg)  X  מינון נדרש (mg/kg) = מ”ג טוטאל', bold: true, size: 24, color: C.primary, rightToLeft: true })],
      })], { bg: C.soft, line: C.primaryLight }));
    } else if (key === 'f2') {
      body.push(oneCellBox([new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [new TextRun({ font: FONT, text: '10 kg X 5 mg/kg = 50 mg (Total)', bold: true, size: 24, color: C.primary })],
      })], { bg: C.soft, line: C.primaryLight }));
    } else {
      const file = key === 'f3' ? 'f3.png' : 'f4.png';
      const sz = pngSize(file);
      const w = key === 'f3' ? 250 : 165;
      body.push(img(file, w, Math.round(w * sz.h / sz.w)));
    }
    body.push(spacer(160));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[proscons\]\s*(.*)$/))) {
    const parts = m[1].split('|').map((s) => s.trim());
    body.push(spacer(80));
    body.push(prosConsBlock(parts[0], parts[1]));
    body.push(spacer(160));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[needles\]\s*(.*)$/))) {
    body.push(spacer(80));
    body.push(needlesBlock(m[1].split('|').map((s) => s.trim())));
    body.push(spacer(180));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[small\]\s*(.*)$/))) {
    body.push(P(m[1], { size: 18, color: C.grayText, alignment: AlignmentType.START, after: 160 }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[tablecaption\]\s*(.*)$/))) {
    body.push(P(m[1], { size: 20, color: C.primary, bold: true, alignment: AlignmentType.START, after: 80, before: 160 }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[table\](.*)$/))) {
    const meta = m[1];
    const wm = meta.match(/widths=([\d,\s]+)/);
    const widths = wm ? wm[1].split(',').map((s) => parseFloat(s.trim())) : null;
    const cm = meta.match(/colors=([0-9A-Fa-f,\s]+)/);
    const colors = cm ? cm[1].split(',').map((s) => s.trim()) : null;
    const rows = [];
    while (i < lines.length && lines[i].trim() !== '[/table]') {
      const r = lines[i].trim();
      if (r.startsWith('|')) {
        rows.push(r.replace(/^\|/, '').replace(/\|$/, '').split('|').map((s) => s.trim()));
      }
      i++;
    }
    i++;
    body.push(spacer(60));
    body.push(tableBlock(widths || rows[0].map(() => 100 / rows[0].length), rows, colors));
    body.push(spacer(200));
    inNum = false; continue;
  }
  if ((m = t.match(/^####\s+(.+)$/))) { body.push(headingPara(m[1], 4)); inNum = false; continue; }
  if ((m = t.match(/^###\s+(.+)$/))) { body.push(headingPara(m[1], 3)); inNum = false; continue; }
  if ((m = t.match(/^##\s+(.+)$/))) { body.push(headingPara(m[1], 2)); inNum = false; continue; }

  if ((m = line.match(/^(\s*)-\s+(.+)$/))) {
    const level = m[1].length >= 2 ? 1 : 0;
    body.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 90, line: 300 },
      numbering: { reference: 'bul', level },
      children: runs(m[2]),
    }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\d+\.\s+(.+)$/))) {
    if (!inNum) { numIdx++; inNum = true; }
    body.push(new Paragraph({
      bidirectional: true, alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 90, line: 300 },
      numbering: { reference: `num-${numIdx % 200}`, level: 0 },
      children: runs(m[1]),
    }));
    continue;
  }

  body.push(P(t, { after: 130 }));
  inNum = false;
}

// ---------- cover ----------
const cover = [
  spacer(1400),
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, spacing: { after: 120 },
    children: [new TextRun({ font: FONT, rightToLeft: true, text: 'רשת "חוות דעת"  •  מחלקת הדמיה', bold: true, size: 22, color: C.primaryLight })],
  }),
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.primary, space: 14 } },
    children: [new TextRun({ font: FONT, rightToLeft: true, text: 'תוכנית הכשרה', bold: true, size: 72, color: C.primary })],
  }),
  spacer(240),
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, spacing: { after: 100 },
    children: [new TextRun({ font: FONT, rightToLeft: true, text: 'לטכנאי הדמיה', bold: true, size: 48, color: C.ink })],
  }),
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
    children: [new TextRun({ font: FONT, rightToLeft: true, text: 'מדריך מקיף לעבודה במחלקת הדמיה – בטיחות, תרופות, הרדמה, ניטור ומצבי חירום', size: 24, color: C.grayText })],
  }),
];

// TOC page
const tocKids = [
  spacer(200),
  new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.primary, space: 8 } },
    children: [new TextRun({ font: FONT, rightToLeft: true, text: 'תוכן העניינים', bold: true, size: 40, color: C.primary })],
  }),
];
let pageMap = {};
try { pageMap = JSON.parse(fs.readFileSync(path.join(DIR, 'pagemap.json'), 'utf8')); } catch (e) {}
toc.forEach((c) => {
  const pg = pageMap[String(c.no)];
  tocKids.push(new Paragraph({
    bidirectional: true, alignment: AlignmentType.START,
    spacing: { after: 60, before: 60, line: 300 },
    border: { bottom: { style: BorderStyle.DOTTED, size: 4, color: C.rule, space: 6 } },
    children: [
      new TextRun({ font: FONT, rightToLeft: true, text: String(c.no).padStart(2, '0') + '\u00A0\u00A0\u00A0\u00A0', bold: true, size: 24, color: C.primaryLight }),
      new TextRun({ font: FONT, rightToLeft: true, text: c.title, bold: true, size: 26, color: C.ink }),
      ...(pg ? [new TextRun({ font: FONT, rightToLeft: true, text: '\u00A0\u00A0\u00A0\u00A0|\u00A0\u00A0\u00A0\u00A0עמ\u0027 ' + pg, size: 22, color: C.grayText })] : []),
    ],
  }));
});

const doc = new Document({
  creator: 'מחלקת הדמיה',
  title: 'תוכנית הכשרה לטכנאי הדמיה',
  numbering,
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: C.body } } },
    characterStyles: [{
      id: 'Hyperlink', name: 'Hyperlink', basedOn: 'DefaultParagraphFont',
      run: { color: C.primaryLight, underline: {} },
    }],
  },
  sections: [
    {
      properties: {
        page: { size: { width: A4_W, height: A4_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
        bidi: true,
      },
      headers: { default: new Header({ children: [new Paragraph({ children: [] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ children: [] })] }) },
      children: [...cover, new Paragraph({ children: [new PageBreak()] }), ...tocKids],
    },
    {
      properties: {
        page: {
          size: { width: A4_W, height: A4_H },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
          pageNumbers: { start: 1 },
        },
        bidi: true,
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            bidirectional: true, alignment: AlignmentType.START,
            spacing: { after: 60 },
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.rule, space: 4 } },
            children: [new TextRun({ font: FONT, rightToLeft: true, text: 'תוכנית הכשרה לטכנאי הדמיה', size: 17, color: C.grayText })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            bidirectional: true, alignment: AlignmentType.CENTER,
            children: [new TextRun({ font: FONT, size: 17, color: C.grayText, children: [PageNumber.CURRENT] })],
          })],
        }),
      },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(DIR, 'out.docx'), buf);
  console.log('wrote out.docx', buf.length, 'bytes; chapters:', toc.length);
});
