# -*- coding: utf-8 -*-
"""Rewrite Section IV of paper_final.docx in IEEE style and insert figures."""
import json, shutil, sys
from copy import deepcopy
import docx
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Pt

SRC = r'C:\Users\lin\Desktop\paper_final.docx'
PRE = r'F:\12.18-2\_emf_preview'          # fig_*.emf live here
crops = json.load(open(PRE + r'\clean\crops.json'))

d = docx.Document(SRC)
body = d.element.body
CH = list(body)

def text_of(ch):
    return ''.join(t.text or '' for t in ch.iter(qn('w:t')))

# ---------- safety checks ----------
expect = {
    42: 'IV Algorithm', 49: 'The SNN branch', 52: 'Unlike conventional',
    57: 'A dedicated binary', 71: 'EN:Because', 76: 'where denotes element-wise',
    85: 'EN:To close', 87: 'where and are sampled', 109: 'EN:A strict',
    124: '6 Conclusion',
}
for i, pref in expect.items():
    assert text_of(CH[i]).strip().startswith(pref), f'anchor {i} mismatch: {text_of(CH[i])[:60]!r}'
DEL_EXPECT = {
    46: 'ZH', 47: '', 48: '其中', 53: 'SNN', 54: '', 55: '其中', 56: '与传统', 58: 'LIF',
    64: '算法', 65: '经式', 66: '对', 67: '经式', 68: '按式', 69: '', 70: '2.3',
    77: 'ZH', 78: '', 79: '其中', 80: '融合在', 81: '', 82: '其中', 83: '', 84: '2.4',
    92: 'ZH', 93: '', 94: '其中', 95: '除静态', 96: '进一步', 97: '', 98: '（满量程',
    103: 'ZH', 104: '', 105: '其中', 106: '训练配方', 107: '', 108: '2.6', 111: 'ZH',
    112: '报告四项', 113: '', 114: '3 Experimental', 115: '实验使用', 118: '4 Results',
    119: '完整模型', 120: '5 Figures', 121: '', 122: '图 1', 123: '图 2',
}
for i, pref in DEL_EXPECT.items():
    got = text_of(CH[i]).strip()
    assert got.startswith(pref), f'DEL anchor {i} mismatch: expect {pref!r}, got {got[:50]!r}'

# ---------- run-level text helpers ----------
def replace_in_p(p_el, old, new, must=True):
    ts = [t for t in p_el.iter(qn('w:t'))]
    for t in ts:
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            return True
    # fallback: phrase spans runs -> merge
    full = ''.join(t.text or '' for t in ts)
    if old in full:
        replaced = full.replace(old, new)
        ts[0].text = replaced
        for t in ts[1:]:
            t.text = ''
        return True
    if must:
        raise AssertionError(f'not found: {old!r} in {full[:80]!r}')
    return False

def set_para_text(p_el, new_text):
    ts = [t for t in p_el.iter(qn('w:t'))]
    assert ts, 'no runs'
    ts[0].text = new_text
    for t in ts[1:]:
        t.text = ''

def append_run(p_el, text, bold=False, sz=22):
    r = parse_xml(
        f'<w:r {nsdecls("w")}><w:rPr>'
        f'<w:rFonts w:ascii="Times New Roman" w:h-ansi="Times New Roman" w:eastAsia="宋体"/>'
        + ('<w:b/>' if bold else '') +
        f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>')
    p_el.append(r)
    return r

def make_para(text_runs, jc=None, space_after=None):
    """text_runs: list of (text, bold, sz). Returns a detached w:p element."""
    p = parse_xml(f'<w:p {nsdecls("w")}><w:pPr>'
                  + (f'<w:jc w:val="{jc}"/>' if jc else '')
                  + '</w:pPr></w:p>')
    for text, bold, sz in text_runs:
        append_run(p, text, bold=bold, sz=sz)
    return p

# ---------- EMF picture insertion ----------
_pkg = d.part.package
def next_image_partname():
    n = 4
    existing = {p.partname for p in _pkg.iter_parts()}
    while PackURI(f'/word/media/image{n}.emf') in existing:
        n += 1
    return PackURI(f'/word/media/image{n}.emf')

_docpr_id = [1000]
EMU_CM = 360000

def emf_drawing(emf_path, fig_key, width_cm):
    blob = open(emf_path, 'rb').read()
    part = Part(next_image_partname(), 'image/x-emf', blob, _pkg)
    rId = d.part.relate_to(part, RT.IMAGE)
    c = crops[fig_key]
    cx = int(width_cm * EMU_CM)
    cy = int(cx * c['aspect'])
    l, t, r, b = (int(c[k] * 100000) for k in ('l', 't', 'r', 'b'))
    _docpr_id[0] += 1
    src_rect = '' if (l == 0 and t == 0 and r == 0 and b == 0) else \
        f'<a:srcRect l="{l}" t="{t}" r="{r}" b="{b}"/>'
    return parse_xml(
        f'<w:r {nsdecls("w", "wp", "a", "pic", "r")}><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{_docpr_id[0]}" name="EMF{_docpr_id[0]}"/>'
        f'<wp:cNvGraphicFramePr/>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{_docpr_id[0]}" name="{fig_key}.emf"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rId}"/>{src_rect}<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')

def fig_para(items):
    """items: list of (emf_name, fig_key, width_cm). Centered paragraph with drawings."""
    p = parse_xml(f'<w:p {nsdecls("w")}><w:pPr><w:jc w:val="center"/></w:pPr></w:p>')
    for i, (emf, key, wcm) in enumerate(items):
        if i:
            append_run(p, '   ', sz=22)
        p.append(emf_drawing(PRE + '\\' + emf, key, wcm))
    return p

def caption(num, text, extra_cn=None):
    runs = [(f'Fig. {num}.  ', True, 20), (text, False, 20)]
    if extra_cn:
        runs.append((extra_cn, False, 20))
    return make_para(runs, jc='center')

# ---------- 1. deletions ----------
DEL = [46, 47, 48, 53, 54, 55, 56, 58, 64, 65, 66, 67, 68, 69, 70,
       77, 78, 79, 80, 81, 82, 83, 84, 92, 93, 94, 95, 96, 97, 98,
       103, 104, 105, 106, 107, 108, 111, 112, 113, 114, 115,
       118, 119, 120, 121, 122, 123]
for i in DEL:
    body.remove(CH[i])

# ---------- 2. text edits ----------
set_para_text(CH[42], 'IV.  Algorithm and Application to Image Recognition')

replace_in_p(CH[49], 'The SNN branch receives',
    'To supply the temporal dynamics that the non-volatile NCZFO device itself lacks (Section III), '
    'an SNN branch is employed, which receives')
replace_in_p(CH[52], 'Please verify the exact surrogate-gradient expression against your code; '
                     'the formula in the revised file appears corrupted.', '')
append_run(CH[57], '  The temporal signal flow of this branch is summarized in Fig. 5.')

replace_in_p(CH[71], 'EN:', '')
replace_in_p(CH[76], '(see Eq. (9))', '(see Eq. (10))')
replace_in_p(CH[85], 'EN:', '')
replace_in_p(CH[85], 'parameterized entirely by measured device data',
    'parameterized entirely by the measured device data of Section III, '
    'namely the HRS/LRS conductance distributions, the EPSC/LTP/LTD switching kinetics, '
    'and the readout noise')
replace_in_p(CH[87], 'the measured HRS.csv and LRS.csv conductance distributions',
    'the measured HRS and LRS conductance distributions')
replace_in_p(CH[109], 'EN:', '')

# table header
tbl = CH[116]
hdr = tbl.findall(qn('w:tr'))[0].findall(qn('w:tc'))
set_para_text(hdr[0].find(qn('w:p')), 'Metric')
set_para_text(hdr[3].find(qn('w:p')), 'Improvement (pp)')
set_para_text(CH[117],
    'TABLE I.  Four-fold LOSO macro-averaged results (mean ± std) of the full '
    'CNN–GRU–SNN model and the no-SNN ablation.')

# ---------- 3. figures after the SNN mechanism (anchor = CH[57]) ----------
anchor = CH[57]
cap5 = caption(5, 'Signal flow of the SNN temporal-processing branch. The CNN feature sequence '
    '(64-D, T = 16) is temporally encoded by the GRU and drives the LIF neuron dynamics '
    '(membrane integration, threshold firing, and reset); the emitted spike sequence is '
    'converted into a fusion gate that modulates the backbone features for the final '
    'ADL/fall decision.')
fig5 = fig_para([('fig_flowchart.emf', 'fig_flowchart', 15.5)])
anchor.addnext(cap5)
anchor.addnext(fig5)

# ---------- 4. results block after the table caption (anchor = CH[117]) ----------
RESULTS = ('The four-fold LOSO results are summarized in Table I and Fig. 6. The full model '
    'outperforms the no-SNN baseline on all four metrics: accuracy and balanced accuracy '
    'improve by 19.49 and 19.33 percentage points (pp), ADL specificity by 7.56 pp, and fall '
    'recall shows the largest gain of 31.10 pp, rising from 43.12% to 74.22%. This indicates '
    'that the pose-event evidence encoded by the LIF population supplies dynamic cues at fall '
    'onset that the CNN–GRU backbone alone cannot capture, while the gated fusion prevents '
    'this evidence from raising false alarms on ADL samples. The per-subject paired comparison '
    'in Fig. 7 shows that the balanced-accuracy gain is positive for every held-out subject '
    'under all three training protocols (Zenodo Only, Recall Priority, and Balanced), '
    'confirming that the improvement is not dominated by any single test subject; since no '
    'significance test has been performed, the gains are reported as consistent macro-averaged '
    'improvements rather than statistically significant differences. Fig. 8 presents the pooled '
    'out-of-fold confusion matrix over all 160 held-out videos, contrasting ideal (I) weight '
    'mapping with the measured non-ideal (NI) device model. Under the non-ideal mapping the '
    'matrix remains diagonally dominant—ADL specificity improves slightly while a limited '
    'fraction of fall recall is traded off—indicating that the measured conductance '
    'variability and readout noise degrade the decision only marginally.')
NOISE = ('Robustness is further evaluated by sweeping the injected noise at inference with the '
    'operating threshold frozen and without any re-tuning. Fig. 9 is reserved for the '
    'degradation of accuracy, ADL specificity, and fall recall under (a) salt-and-pepper, '
    '(b) Gaussian, and (c) Poisson input noise; these experiments are currently being repeated.')

anchor = CH[117]
block = []
block.append(make_para([(RESULTS, False, 22)]))
block.append(fig_para([('fig_bar.emf', 'fig_bar', 8.2), ('fig_delta.emf', 'fig_delta', 8.2)]))
block.append(caption(6, 'Four-fold LOSO comparison of the CNN–GRU baseline (no-SNN) and the full '
    'CNN–GRU–SNN model: (a) macro-averaged accuracy, balanced accuracy, ADL specificity, and '
    'fall recall; (b) improvement of the full model over the baseline in percentage points.'))
block.append(fig_para([('fig_pair1.emf', 'fig_pair1', 5.4), ('fig_pair2.emf', 'fig_pair2', 5.4),
                       ('fig_pair3.emf', 'fig_pair3', 5.4)]))
block.append(caption(7, 'Per-subject paired comparison of balanced accuracy between the no-SNN '
    'baseline and the full model across the four LOSO folds (S1–S4) under three training '
    'protocols: (a) Zenodo Only, (b) Recall Priority, and (c) Balanced. The orange diamond '
    'marks the fold mean.'))
block.append(fig_para([('fig_ideal.emf', 'fig_ideal', 9.5)]))
block.append(caption(8, 'Pooled out-of-fold confusion matrix over all 160 held-out videos with '
    'ideal (I) and measured non-ideal (NI) crossbar weight mapping.'))
block.append(make_para([(NOISE, False, 22)]))
for lab in ('(a) salt-and-pepper', '(b) Gaussian', '(c) Poisson'):
    block.append(make_para([(f'[ 待插入图：Fig. 9 {lab} 噪声下的 Accuracy / ADL specificity / '
                             f'Fall recall 鲁棒性曲线（实验重跑中）]', False, 22)], jc='center'))
block.append(caption(9, 'Robustness of the full model against input noise with the operating '
    'threshold frozen: (a) salt-and-pepper; (b) Gaussian; (c) Poisson.',
    extra_cn='  [占位：三张噪声图重跑完成后插入]'))
for el in reversed(block):
    anchor.addnext(el)

# ---------- 5. conclusion ----------
p = CH[124]
ppr = p.find(qn('w:pPr'))
if ppr is not None:
    st = ppr.find(qn('w:pStyle'))
    if st is not None:
        ppr.remove(st)
set_para_text(p, 'V.  Conclusion')
for r in p.findall(qn('w:r')):
    rpr = r.find(qn('w:rPr'))
    if rpr is None:
        rpr = parse_xml(f'<w:rPr {nsdecls("w")}/>')
        r.insert(0, rpr)
    if rpr.find(qn('w:b')) is None:
        rpr.append(parse_xml(f'<w:b {nsdecls("w")}/>'))
    for tag in ('w:sz', 'w:szCs'):
        e = rpr.find(qn(tag))
        if e is None:
            e = parse_xml(f'<{tag} {nsdecls("w")} w:val="22"/>')
            rpr.append(e)
        else:
            e.set(qn('w:val'), '22')

CONCL = ('This work has established a memristive-crossbar-oriented hybrid CNN–SNN method for '
    'fall detection. A CNN–GRU backbone extracts spatial and short-term temporal features from '
    'video, a pose-event-driven LIF-SNN branch supplies complementary evidence of fall '
    'dynamics, and an event-aware group–channel gate controls how strongly this temporal '
    'evidence modulates the backbone decision. Under a strict four-fold LOSO protocol, the '
    'full model improves all four macro-averaged metrics over the CNN–GRU no-SNN baseline, '
    'with the largest gain in fall recall. Remaining work includes completing the PLD process '
    'parameters, the statistical device characterization, the PPF measurement, and the ongoing '
    'input-noise experiments. The device data in this work were measured on single devices '
    'with dot top electrodes, and the crossbar-level results were obtained from simulations '
    'parameterized by these measurements. Upon integration toward patterned crossbar arrays, '
    'the patterned bottom electrode forms a height step comparable to its own thickness at '
    'the crosspoint edge (bottom-electrode thickness plus film thickness at the crosspoint, '
    'versus film thickness only in the neighboring region), over which the functional film '
    'covers discontinuously and introduces edge-leakage paths between the top and bottom '
    'electrodes; this motivates the present choice of dot-electrode single-device measurements '
    'combined with measurement-based array-level modeling. Process optimizations such as '
    'thinning the bottom electrode, adding a planarization layer, or adopting recessed bottom '
    'electrodes are left to future work.')
set_para_text(CH[125], CONCL)

d.save(SRC)
print('SAVED OK')
