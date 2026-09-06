const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, PageBreak,
  Header, Footer, PageNumber, LevelFormat, ExternalHyperlink, VerticalAlign,
  convertMillimetersToTwip, TableLayoutType, HeightRule,
} = d;

const DIR = __dirname;
const HF = 'Heebo';      // headings
const BF = 'Assistant';  // body

// ---------- palette ----------
const C = {
  navy: '1B365D', navyDeep: '132A4A',
  teal: '1F8A8A', tealDark: '166A6A', tealLight: 'A7DADC', tealPale: 'E4F2F2',
  ink: '1F2933', body: '2B3945', gray: '6E7C87', rule: 'D6DEE6',
  soft: 'EEF4F7', softer: 'F6F9FB', white: 'FFFFFF',
  dangerLine: 'C0392B', dangerBg: 'FDECEA', dangerText: '8E2B23',
  warnLine: 'D98E04', warnBg: 'FFF5E0', warnText: '7A5000',
  noteLine: '1F8A8A', noteBg: 'E8F4F4', noteText: '166A6A',
  tipLine: '3C8D5A', tipBg: 'EBF6EE', tipText: '2B6A43',
  prosBg: 'EDF7F0', prosLine: '3C8D5A', consBg: 'FDEFEC', consLine: 'C0392B',
};

const A4_W = 11906, A4_H = 16838;
const MARGIN = convertMillimetersToTwip(20);
const CONTENT_W = A4_W - MARGIN * 2;

// ---------- helpers ----------
const noBorder = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder,
  insideHorizontal: noBorder, insideVertical: noBorder };
const line = (color, size) => ({ style: BorderStyle.SINGLE, size, color });

function runs(text, opts = {}) {
  const base = { font: opts.font || BF, rightToLeft: true, size: opts.size || 23, color: opts.color || C.body };
  const out = [];
  String(text).split('**').forEach((p, i) => {
    if (p === '') return;
    out.push(new TextRun({ ...base, text: p, bold: opts.bold || (i % 2 === 1) }));
  });
  if (!out.length) out.push(new TextRun({ ...base, text: '' }));
  return out;
}

function P(text, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    alignment: opts.alignment || AlignmentType.START,
    spacing: { before: opts.before || 0, after: opts.after === undefined ? 140 : opts.after, line: opts.line || 336 },
    keepNext: opts.keepNext,
    border: opts.border,
    children: opts.children || runs(text, opts),
  });
}
const spacer = (h) => new Paragraph({ spacing: { before: 0, after: h || 120 }, children: [] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function img(file, wPt, hPt) {
  return new Paragraph({
    bidirectional: true, alignment: AlignmentType.CENTER, spacing: { before: 100, after: 80 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(path.join(DIR, file)), transformation: { width: wPt, height: hPt } })],
  });
}
function pngSize(file) { const b = fs.readFileSync(path.join(DIR, file)); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function box(children, { bg, bar, pad = 200, barSize = 30 }) {
  return new Table({
    visuallyRightToLeft: true,
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [CONTENT_W], layout: TableLayoutType.FIXED, borders: noBorders,
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      children, width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: bg, color: 'auto' },
      borders: { top: line(bg, 2), bottom: line(bg, 2), right: line(bg, 2), left: bar ? line(bar, barSize) : line(bg, 2) },
      margins: { top: pad - 40, bottom: pad - 40, left: pad + 60, right: pad + 60 },
    })] })],
  });
}

// ---------- numbering ----------
const numbering = { config: [] };
for (let i = 0; i < 200; i++) {
  numbering.config.push({ reference: `num-${i}`, levels: [{
    level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.START,
    style: { paragraph: { indent: { start: 500, hanging: 320 } }, run: { font: HF, color: C.teal, bold: true } },
  }] });
}
numbering.config.push({ reference: 'bul', levels: [
  { level: 0, format: LevelFormat.BULLET, text: '■', alignment: AlignmentType.START,
    style: { paragraph: { indent: { start: 440, hanging: 280 } }, run: { font: 'Arial', color: C.teal, size: 13 } } },
  { level: 1, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.START,
    style: { paragraph: { indent: { start: 860, hanging: 280 } }, run: { font: 'Arial', color: C.teal, size: 20 } } },
] });

// ---------- parse content ----------
const CHANGES = !!process.env.CHANGES;
const files = CHANGES ? ['changes.txt'] : ['part1.txt', 'part2.txt', 'part3.txt', 'part4.txt', 'part5.txt'];
let lines = [];
for (const f of files) lines = lines.concat(fs.readFileSync(path.join(DIR, 'content', f), 'utf8').split('\n'));

let pageMap = {};
try { pageMap = JSON.parse(fs.readFileSync(path.join(DIR, 'pagemap.json'), 'utf8')); } catch (e) {}

const body = [];
const toc = [];           // {level:1|2, title, no}
let numIdx = 0, inNum = false, chapterNo = 0, firstDrug = true, lastCallout = null;

function heading(text, level) {
  if (level === 2) {
    toc.push({ level: 2, title: text });
    return new Paragraph({
      bidirectional: true, alignment: AlignmentType.START, keepNext: true,
      spacing: { before: 400, after: 160, line: 300 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.tealLight, space: 6 } },
      children: [new TextRun({ font: HF, rightToLeft: true, text, bold: true, size: 30, color: C.navy })],
    });
  }
  if (level === 3) {
    return new Paragraph({
      bidirectional: true, alignment: AlignmentType.START, keepNext: true,
      spacing: { before: 300, after: 100, line: 300 },
      children: [new TextRun({ font: HF, rightToLeft: true, text, bold: true, size: 25, color: C.tealDark })],
    });
  }
  return new Paragraph({
    bidirectional: true, alignment: AlignmentType.START, keepNext: true,
    spacing: { before: 220, after: 80, line: 300 },
    children: [new TextRun({ font: HF, rightToLeft: true, text, bold: true, size: 23, color: C.ink })],
  });
}

function chapterOpener(title, label) {
  if (!label) { chapterNo += 1; toc.push({ level: 1, title, no: chapterNo }); }
  const n = String(chapterNo).padStart(2, '0');
  const pre = (chapterNo <= 1 || label) ? [] : [pageBreak()];
  return [
    ...pre,
    new Table({
      visuallyRightToLeft: true,
      width: { size: 100, type: WidthType.PERCENTAGE },
      columnWidths: [CONTENT_W], layout: TableLayoutType.FIXED, borders: noBorders,
      rows: [
        new TableRow({ cantSplit: true, height: { value: 2300, rule: HeightRule.ATLEAST }, children: [new TableCell({
          width: { size: CONTENT_W, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: C.navy, color: 'auto' },
          borders: noBorders, verticalAlign: VerticalAlign.CENTER,
          margins: { top: 240, bottom: 240, left: 360, right: 360 },
          children: [
            new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 60 },
              children: [new TextRun({ font: HF, rightToLeft: true, text: label || `פרק ${n}`, bold: true, size: 22, color: C.tealLight })] }),
            new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
              children: [new TextRun({ font: HF, rightToLeft: true, text: title, bold: true, size: 52, color: C.white })] }),
          ],
        })] }),
        new TableRow({ cantSplit: true, height: { value: 160, rule: HeightRule.EXACT }, children: [new TableCell({
          width: { size: CONTENT_W, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: C.teal, color: 'auto' },
          borders: noBorders, margins: { top: 0, bottom: 0, left: 0, right: 0 },
          children: [new Paragraph({ spacing: { after: 0, line: 100 }, children: [] })],
        })] }),
      ],
    }),
    spacer(320),
  ];
}

function callout(kind, title, text, extra = []) {
  const m = {
    danger: { bg: C.dangerBg, bar: C.dangerLine, tc: C.dangerText },
    warn: { bg: C.warnBg, bar: C.warnLine, tc: C.warnText },
    note: { bg: C.noteBg, bar: C.noteLine, tc: C.noteText },
    tip: { bg: C.tipBg, bar: C.tipLine, tc: C.tipText },
  }[kind];
  const kids = [];
  if (title) kids.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: text ? 70 : 0, line: 300 },
    children: [new TextRun({ font: HF, rightToLeft: true, text: title, bold: true, size: 24, color: m.tc })] }));
  if (text) kids.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: extra.length ? 110 : 0, line: 330 },
    children: runs(text, { size: 23, color: m.tc }) }));
  extra.forEach((x, k) => kids.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: k === extra.length - 1 ? 0 : 110, line: 330 },
    children: runs(x, { size: 23, color: m.tc }) })));
  return box(kids, { bg: m.bg, bar: m.bar });
}

function drugCard(name, short, eng, routes) {
  const kids = [
    new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 30 }, keepNext: true,
      children: [
        new TextRun({ font: HF, rightToLeft: true, text: name, bold: true, size: 40, color: C.navy }),
        ...(short ? [new TextRun({ font: HF, rightToLeft: true, text: `  (${short})`, size: 26, color: C.gray })] : []),
      ] }),
    new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 90 }, keepNext: true,
      children: [new TextRun({ font: HF, rightToLeft: true, text: eng, size: 24, color: C.tealDark })] }),
    new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 }, keepNext: true,
      children: [
        new TextRun({ font: BF, rightToLeft: true, text: 'דרכי מתן:  ', size: 22, color: C.gray }),
        new TextRun({ font: HF, rightToLeft: true, text: routes.split('/').map((s) => s.trim()).join('   ·   '), bold: true, size: 23, color: C.navy }),
      ] }),
  ];
  return box(kids, { bg: C.soft, bar: C.teal, pad: 260, barSize: 40 });
}

function prosCons(pros, cons) {
  const half = Math.floor(CONTENT_W / 2) - 60;
  const gap = CONTENT_W - half * 2;
  const col = (t, txt, bg, ln) => new TableCell({
    width: { size: half, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: bg, color: 'auto' },
    borders: { top: line(ln, 14), bottom: line(bg, 2), left: line(bg, 2), right: line(bg, 2) },
    margins: { top: 150, bottom: 160, left: 200, right: 200 },
    children: [
      new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 60 },
        children: [new TextRun({ font: HF, rightToLeft: true, text: t, bold: true, size: 22, color: ln })] }),
      new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0, line: 320 }, children: runs(txt, { size: 22 }) }),
    ],
  });
  const gapCell = new TableCell({ width: { size: gap, type: WidthType.DXA }, borders: noBorders, children: [new Paragraph({ children: [] })] });
  return new Table({
    visuallyRightToLeft: true, width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [half, gap, half], layout: TableLayoutType.FIXED, borders: noBorders,
    rows: [new TableRow({ cantSplit: true, children: [col('יתרונות', pros, C.prosBg, C.prosLine), gapCell, col('חסרונות', cons, C.consBg, C.consLine)] })],
  });
}

function needles(items) {
  const sw = { 'כתומה': 'E8862E', 'כחולה': '3B7FC4', 'ירוקה': '3FA75C', 'וורודה': 'D4649B' };
  const thick = [6, 16, 28, 44];               // bottom bar grows: thin -> thick
  const chipW = Math.floor(CONTENT_W * 0.19), arrowW = Math.floor((CONTENT_W - chipW * items.length) / (items.length - 1));
  const cols = []; items.forEach((it, k) => { cols.push(chipW); if (k < items.length - 1) cols.push(arrowW); });
  const chip = (it, k) => new TableCell({
    width: { size: chipW, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: sw[it] || 'CCCCCC', color: 'auto' },
    borders: { top: noBorder, left: noBorder, right: noBorder, bottom: line(C.navy, thick[k] || 20) },
    margins: { top: 150, bottom: 150, left: 60, right: 60 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ font: HF, rightToLeft: true, text: it, bold: true, size: 23, color: 'FFFFFF' })] })],
  });
  const arrow = () => new TableCell({
    width: { size: arrowW, type: WidthType.DXA }, borders: noBorders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 0, bottom: 0, left: 0, right: 0 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ font: 'Arial', text: '←', bold: true, size: 30, color: C.teal })] })],
  });
  const label = (txt, w, align) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders: noBorders, margins: { top: 60, bottom: 0, left: 0, right: 0 },
    children: [new Paragraph({ bidirectional: true, alignment: align, spacing: { after: 0 },
      children: [new TextRun({ font: HF, rightToLeft: true, text: txt, bold: true, size: 19, color: C.gray })] })],
  });
  const row1 = []; items.forEach((it, k) => { row1.push(chip(it, k)); if (k < items.length - 1) row1.push(arrow()); });
  const row2 = [label('הכי דקה', chipW, AlignmentType.START)];
  for (let k = 1; k < cols.length - 1; k++) row2.push(new TableCell({ width: { size: cols[k], type: WidthType.DXA }, borders: noBorders, children: [new Paragraph({ children: [] })] }));
  row2.push(label('הכי עבה', chipW, AlignmentType.END));
  return new Table({
    visuallyRightToLeft: true, width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: cols, layout: TableLayoutType.FIXED, borders: noBorders,
    rows: [new TableRow({ cantSplit: true, children: row1 }), new TableRow({ cantSplit: true, children: row2 })],
  });
}

function table(widths, rows, colors) {
  const cols = widths.map((p) => Math.floor(CONTENT_W * p / 100));
  return new Table({
    visuallyRightToLeft: true, width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: cols, layout: TableLayoutType.FIXED, borders: noBorders,
    rows: rows.map((cells, ri) => {
      const head = ri === 0;
      return new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((txt, ci) => {
        let fill = head ? C.navy : (ri % 2 === 0 ? C.softer : C.white);
        let color = head ? C.white : C.body;
        let bold = head;
        if (!head && colors && ci === 0 && colors[ri - 1]) { fill = colors[ri - 1]; color = C.white; bold = true; }
        return new TableCell({
          width: { size: cols[ci], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill, color: 'auto' },
          borders: { top: line(head ? C.navy : C.rule, 4), bottom: line(head ? C.navy : C.rule, 4), left: line(fill, 2), right: line(fill, 2) },
          margins: { top: 130, bottom: 130, left: 170, right: 170 }, verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0, line: 300 },
            children: runs(txt, { size: 22, color, bold, font: head ? HF : BF }) })],
        });
      }) });
    }),
  });
}

// ---------- main loop ----------
let i = 0;
while (i < lines.length) {
  const raw = lines[i]; const t = raw.trim(); i++;
  if (t === '') { inNum = false; lastCallout = null; continue; }
  if (t === '[steps]' || t === '[/steps]') continue;
  let m;
  if ((m = t.match(/^\[chapter\]\s*(.+)$/))) { body.push(...chapterOpener(m[1].trim())); inNum = false; continue; }
  if ((m = t.match(/^\[doctitle\]\s*(.+)$/))) { body.push(...chapterOpener(m[1].trim(), 'נספח לתוכנית ההכשרה לטכנאי הדמיה')); inNum = false; continue; }
  if ((m = t.match(/^\[final\]\s*(.+)$/))) {
    body.push(pageBreak(), spacer(1800));
    body.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ font: HF, rightToLeft: true, text: m[1].trim(), bold: true, size: 64, color: C.dangerLine })] }));
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 320 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.dangerLine, space: 1 } }, indent: { left: 3600, right: 3600 }, children: [] }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[drug\]\s*(.*)$/))) {
    const p = m[1].split('|').map((s) => s.trim());
    if (!firstDrug) body.push(pageBreak()); firstDrug = false;
    body.push(spacer(80), drugCard(p[0], p[1], p[2], p[3]), spacer(120)); inNum = false; continue;
  }
  if ((m = t.match(/^\[(danger|warn|note|tip)\]\s*(.*)$/))) {
    let title = null, text = m[2];
    if (text.includes('|')) { const k = text.indexOf('|'); title = text.slice(0, k).trim(); text = text.slice(k + 1).trim().replace(/^[-–]\s*/, ''); }
    const extra = [];
    while (i < lines.length && lines[i].trim().startsWith('> ')) { extra.push(lines[i].trim().slice(2)); i++; }
    body.push(spacer(120), callout(m[1], title, text, extra), spacer(180)); lastCallout = m[1]; inNum = false; continue;
  }
  if ((m = t.match(/^\[img\]\s*(.*)$/))) {
    const p = m[1].split('|').map((s) => s.trim());
    if (p[1]) body.push(P(p[1], { after: 40 }));
    const sz = pngSize(p[0]); const h = 96;
    body.push(img(p[0], Math.round(h * sz.w / sz.h), h));
    if (p[2]) body.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new ExternalHyperlink({ link: p[2], children: [new TextRun({ font: BF, text: 'לחצו כאן לצפייה בסרטון', style: 'Hyperlink', size: 20, rightToLeft: true })] })] }));
    inNum = false; continue;
  }
  if ((m = t.match(/^\[formula\]\s*(.+)$/))) {
    const key = m[1].trim();
    if (key === 'f1' || key === 'f2') {
      const txt = key === 'f1' ? 'משקל החיה (kg)  X  מינון נדרש (mg/kg) = מ”ג טוטאל' : '10 kg X 5 mg/kg = 50 mg (Total)';
      body.push(box([new Paragraph({ bidirectional: key === 'f1', alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [new TextRun({ font: HF, rightToLeft: key === 'f1', text: txt, bold: true, size: 26, color: C.navy })] })], { bg: C.tealPale, bar: null, pad: 200 }));
    } else {
      const file = key === 'f3' ? 'f3.png' : 'f4.png'; const sz = pngSize(file); const w = key === 'f3' ? 250 : 165;
      body.push(img(file, w, Math.round(w * sz.h / sz.w)));
    }
    body.push(spacer(160)); inNum = false; continue;
  }
  if ((m = t.match(/^\[proscons\]\s*(.*)$/))) { const p = m[1].split('|').map((s) => s.trim()); body.push(spacer(100), prosCons(p[0], p[1]), spacer(200)); inNum = false; continue; }
  if ((m = t.match(/^\[needles\]\s*(.*)$/))) { body.push(spacer(80), needles(m[1].split('|').map((s) => s.trim())), spacer(200)); inNum = false; continue; }
  if ((m = t.match(/^\[small\]\s*(.*)$/))) { body.push(P(m[1], { size: 19, color: C.gray, after: 180 })); inNum = false; continue; }
  if ((m = t.match(/^\[tablecaption\]\s*(.*)$/))) { body.push(P(m[1], { size: 22, color: C.navy, bold: true, font: HF, after: 90, before: 200, keepNext: true })); inNum = false; continue; }
  if ((m = t.match(/^\[table\](.*)$/))) {
    const wm = m[1].match(/widths=([\d,\s]+)/); const cm = m[1].match(/colors=([0-9A-Fa-f,\s]+)/);
    const rows = [];
    while (i < lines.length && lines[i].trim() !== '[/table]') { const r = lines[i].trim(); if (r.startsWith('|')) rows.push(r.replace(/^\|/, '').replace(/\|$/, '').split('|').map((s) => s.trim())); i++; }
    i++;
    body.push(spacer(80), table(wm ? wm[1].split(',').map(Number) : rows[0].map(() => 100 / rows[0].length), rows, cm ? cm[1].split(',').map((s) => s.trim()) : null), spacer(240));
    inNum = false; continue;
  }
  if ((m = t.match(/^####\s+(.+)$/))) { body.push(heading(m[1], 4)); inNum = false; continue; }
  if ((m = t.match(/^###\s+(.+)$/))) { body.push(heading(m[1], 3)); inNum = false; continue; }
  if ((m = t.match(/^##\s+(.+)$/))) { body.push(heading(m[1], 2)); inNum = false; continue; }
  if ((m = raw.match(/^(\s*)-\s+(.+)$/))) {
    body.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 100, line: 330 },
      numbering: { reference: 'bul', level: m[1].length >= 2 ? 1 : 0 }, children: runs(m[2]) }));
    continue;
  }
  if ((m = t.match(/^\d+\.\s+(.+)$/))) {
    if (!inNum) { numIdx++; inNum = true; }
    body.push(new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 100, line: 330 },
      numbering: { reference: `num-${numIdx % 200}`, level: 0 }, children: runs(m[1]) }));
    continue;
  }
  body.push(P(t)); inNum = false;
}

// ---------- cover ----------
const cover = [
  new Table({
    visuallyRightToLeft: true, width: { size: A4_W, type: WidthType.DXA },
    columnWidths: [A4_W], layout: TableLayoutType.FIXED, borders: noBorders,
    rows: [
      new TableRow({ height: { value: 9800, rule: HeightRule.ATLEAST }, children: [new TableCell({
        width: { size: A4_W, type: WidthType.DXA }, borders: noBorders,
        shading: { type: ShadingType.CLEAR, fill: C.navy, color: 'auto' },
        margins: { top: 0, bottom: 0, left: 1300, right: 1300 },
        children: [
          new Paragraph({ spacing: { before: 5200, after: 0 }, children: [] }),
          new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 320 },
            children: [new TextRun({ font: HF, rightToLeft: true, text: 'רשת "חוות דעת"   •   מחלקת הדמיה', bold: true, size: 24, color: C.tealLight })] }),
          new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 120 },
            children: [new TextRun({ font: HF, rightToLeft: true, text: 'תוכנית הכשרה', bold: true, size: 100, color: C.white })] }),
          new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
            children: [new TextRun({ font: HF, rightToLeft: true, text: 'לטכנאי הדמיה', bold: true, size: 68, color: C.tealLight })] }),
        ],
      })] }),
      new TableRow({ height: { value: 260, rule: HeightRule.EXACT }, children: [new TableCell({
        width: { size: A4_W, type: WidthType.DXA }, borders: noBorders,
        shading: { type: ShadingType.CLEAR, fill: C.teal, color: 'auto' },
        margins: { top: 0, bottom: 0, left: 0, right: 0 },
        children: [new Paragraph({ spacing: { after: 0, line: 120 }, children: [] })],
      })] }),
    ],
  }),
  spacer(900),
  new Paragraph({ bidirectional: true, alignment: AlignmentType.START, indent: { start: 1300, end: 1300 }, spacing: { after: 160 },
    children: [new TextRun({ font: HF, rightToLeft: true, text: 'מדריך מקיף לעבודה במחלקת ההדמיה', bold: true, size: 32, color: C.navy })] }),
  new Paragraph({ bidirectional: true, alignment: AlignmentType.START, indent: { start: 1300, end: 1300 }, spacing: { after: 0, line: 330 },
    children: [new TextRun({ font: BF, rightToLeft: true, text: 'בטיחות  ·  תרופות  ·  בדיקות דם  ·  הרדמה  ·  ניטור  ·  מצבי חירום', size: 26, color: C.gray })] }),
];

// ---------- TOC ----------
const tocKids = [
  spacer(120),
  new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 260 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: C.teal, space: 10 } },
    children: [new TextRun({ font: HF, rightToLeft: true, text: 'תוכן העניינים', bold: true, size: 48, color: C.navy })] }),
];
const tocRows = [];
const wTitle = Math.floor(CONTENT_W * 0.86), wPage = CONTENT_W - wTitle;
toc.forEach((e) => {
  const key = e.level === 1 ? `c${e.no}` : `h:${e.title}`;
  const pg = pageMap[key];
  const isCh = e.level === 1;
  tocRows.push(new TableRow({ cantSplit: true, children: [
    new TableCell({
      width: { size: wTitle, type: WidthType.DXA },
      borders: { top: noBorder, left: noBorder, right: noBorder, bottom: isCh ? noBorder : { style: BorderStyle.DOTTED, size: 4, color: C.rule } },
      margins: { top: isCh ? 150 : 38, bottom: isCh ? 40 : 38, left: 0, right: isCh ? 0 : 420 },
      children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
        children: isCh ? [
          new TextRun({ font: HF, rightToLeft: true, text: String(e.no).padStart(2, '0') + '   ', bold: true, size: 26, color: C.teal }),
          new TextRun({ font: HF, rightToLeft: true, text: e.title, bold: true, size: 27, color: C.navy }),
        ] : [new TextRun({ font: BF, rightToLeft: true, text: e.title, size: 21, color: C.body })] })],
    }),
    new TableCell({
      width: { size: wPage, type: WidthType.DXA },
      borders: { top: noBorder, left: noBorder, right: noBorder, bottom: isCh ? noBorder : { style: BorderStyle.DOTTED, size: 4, color: C.rule } },
      margins: { top: isCh ? 150 : 38, bottom: isCh ? 40 : 38, left: 0, right: 0 }, verticalAlign: VerticalAlign.BOTTOM,
      children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.END, spacing: { after: 0 },
        children: [new TextRun({ font: HF, rightToLeft: true, text: pg ? String(pg) : '', bold: isCh, size: isCh ? 26 : 22, color: isCh ? C.teal : C.gray })] })],
    }),
  ] }));
});
tocKids.push(new Table({ visuallyRightToLeft: true, width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [wTitle, wPage], layout: TableLayoutType.FIXED, borders: noBorders, rows: tocRows }));
if (!CHANGES) fs.writeFileSync(path.join(DIR, 'toc.json'), JSON.stringify(toc));

const emptyHF = { default: new Header({ children: [new Paragraph({ children: [] })] }) };
const emptyFF = { default: new Footer({ children: [new Paragraph({ children: [] })] }) };
const pageProps = (m) => ({ size: { width: A4_W, height: A4_H }, margin: { top: m, bottom: m, left: m, right: m, header: 500, footer: 500 } });

const doc = new Document({
  creator: 'מחלקת הדמיה', title: 'תוכנית הכשרה לטכנאי הדמיה',
  numbering,
  styles: {
    default: { document: { run: { font: BF, size: 23, color: C.body } } },
    characterStyles: [{ id: 'Hyperlink', name: 'Hyperlink', basedOn: 'DefaultParagraphFont', run: { color: C.teal, underline: {} } }],
  },
  sections: [
    ...(CHANGES ? [] : [
    { properties: { page: pageProps(0) }, headers: emptyHF, footers: emptyFF, children: cover },
    { properties: { page: pageProps(MARGIN) }, headers: emptyHF, footers: emptyFF, children: tocKids }]),
    {
      properties: { page: { ...pageProps(MARGIN), pageNumbers: { start: 1 } } },
      headers: { default: new Header({ children: [new Paragraph({
        bidirectional: true, alignment: AlignmentType.START, spacing: { after: 0 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.rule, space: 6 } },
        children: [new TextRun({ font: HF, rightToLeft: true, text: 'תוכנית הכשרה לטכנאי הדמיה', size: 18, color: C.gray }),
                   new TextRun({ font: HF, rightToLeft: true, text: '   |   רשת "חוות דעת"  ·  מחלקת הדמיה', size: 18, color: 'A0ACB5' })],
      })] }) },
      footers: { default: new Footer({ children: [new Paragraph({
        bidirectional: true, alignment: AlignmentType.CENTER,
        children: [new TextRun({ font: HF, size: 19, color: C.teal, bold: true, children: [PageNumber.CURRENT] })],
      })] }) },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(DIR, CHANGES ? 'changes.docx' : 'out.docx'), buf);
  console.log('wrote docx', buf.length, 'bytes; toc entries:', toc.length);
});
