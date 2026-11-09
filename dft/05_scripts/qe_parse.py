#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QE (7.3) 输出解析核心库 —— 全项目唯一权威实现。

输入
----
- pw.x 文本输出（calculation = scf / relax / vc-relax / md / nscf），通常为 *.out；
  也接受已读入的字符串（用 text= 参数，便于单元测试）。
- neb.x 文本输出（含 image 表 / activation energy / path length），见 parse_neb_out()。

输出
----
纯 Python dict（不依赖 numpy / pandas），字段见各函数 docstring。
所有解析都**以「计算块」为最小单位**：一个块 = 一次 SCF 自洽求解。
块内 [F, -TS, E=F+TS, 磁矩, 力, 收敛状态] 必须同属一块，禁止各自取全文最后值再拼。

单位约定（全项目统一，禁止在别处另写常量）
------------------------------------------
- 能量：QE 打印为 Ry。1 Ry = 13.605693122994 eV            (RY_TO_EV)
- 力  ：QE 打印为 Ry/bohr。1 Ry/bohr = 25.7110456 eV/Å     (RY_BOHR_TO_EV_A)
        （1 bohr = 0.529177210903 Å；13.605693122994/0.529177210903 = 25.7110456）
- 长度：1 bohr = 0.529177210903 Å                          (BOHR_TO_A)
- 磁矩：QE 直接打印 μB（Bohr mag/cell 或 每原子 magn），不换算。
- 展宽：QE 打印 "smearing contrib. (-TS) = x Ry"，即 **x = -TS**。
        本库同时保存 minus_TS_ry = x 与 ts_ry = -x；E = F + TS = F - minus_TS。

两个容易混的量（务必分开，禁止互相替代）
----------------------------------------
- max_force  ：max_i ||F_i||   —— 单个原子的最大力矢量模（判断收敛/残差用它）。
- total_force："Total force = X" —— 全体原子力的范数 sqrt(sum_i |F_i|^2)，是整体量，
               一般比 max_force 大（55 原子时约 5~7 倍），**不是**最大原子力。

假设
----
1. 块起点是 "Self-consistent Calculation" 行（relax/md 每个离子步一行，NEB 每个 image
   一行）；若文件里没有该行（纯 scf 也可能有），退化为 "iteration #  1" 行。
   块终点 = 下一个块起点。
2. verbosity='high' 时 "Forces acting on atoms ..." 之后有 7 个 nat 行子表：
   总力 / non-local / ionic / local / core correction / Hubbard / SCF correction。
   子表之间没有重复的标题行，只靠 "The ... to forces" 文字行分隔。
   本库只取**第一个**子表（总力），并显式忽略其余 6 个。
3. "convergence has been achieved in N iterations" / "convergence NOT achieved after N
   iterations" 只出现在块内、且在最终 `!` 能量之后；块内缺失该语句 = 不能认定收敛。
4. 文件是否正常结束（JOB DONE）与 SCF 是否收敛是两个独立事实，分别记录。

自测 / 回归见 04_分析脚本\\tests\\run_tests.py。
"""

import hashlib
import json
import math
import os
import re
import time

# --------------------------------------------------------------------------- #
# 统一单位换算常量（改这里就等于改全项目；历史上 25.7110336/25.711043/25.711033
# 三个不一致的值已统一为本文件的一组）
# --------------------------------------------------------------------------- #
RY_TO_EV = 13.605693122994          # 1 Ry  -> eV   (QE 内部转换因子)
BOHR_TO_A = 0.529177210903          # 1 bohr -> Å
RY_BOHR_TO_EV_A = 25.7110456        # 1 Ry/bohr -> eV/Å

# 兼容旧字段名（其它脚本/笔记里出现过这些名字）
EV_PER_RY = RY_TO_EV
EV_PER_ANG = RY_BOHR_TO_EV_A

# QE 打印的浮点数：允许 1.0E-06 / 0.00000000 / -8817.29148861 / D 指数
_F = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?'

RE_QE_FLOAT = _F


class QEParseError(RuntimeError):
    """解析/口径错误基类。"""


class QEConvergenceError(QEParseError):
    """最后一个计算块未收敛，却被当作最终结果使用。"""


class SetupMismatchError(QEParseError):
    """两次能量的计算设置（截断能/k 点/展宽/U/赝势/能量定义/几何）不一致。"""


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _f(s):
    """QE 风格浮点串 -> float（容忍 Fortran 的 D 指数）。"""
    if s is None:
        return None
    return float(s.replace('D', 'E').replace('d', 'e'))


def read_text(path):
    """读入文本（容忍编码问题，不因单字节错误失败）。"""
    with open(path, 'r', errors='replace') as f:
        return f.read()


def _last(regex, text, flags=0):
    ms = list(re.finditer(regex, text, flags))
    return ms[-1] if ms else None


def _all_float(regex, text, flags=0):
    return [_f(m.group(1)) for m in re.finditer(regex, text, flags)]


# --------------------------------------------------------------------------- #
# 计算块切分
# --------------------------------------------------------------------------- #
RE_BLOCK_SELF = re.compile(r'^[ \t]*Self-consistent Calculation[ \t]*$', re.M)
RE_BLOCK_ITER1 = re.compile(r'^[ \t]*iteration #\s*1[ \t]+ecut=', re.M)
RE_END_SCF = re.compile(r'^[ \t]*End of self-consistent calculation[ \t]*$', re.M)


def split_calc_blocks(text):
    """把 pw.x 输出切成「一次 SCF 自洽求解」块。

    返回 [{'index':1-based, 'label':..., 'start':int, 'end':int, 'text':str}, ...]
    若文件里没有任何块起点（如被截断在 header 之后），返回 []。
    """
    anchors = []
    if RE_BLOCK_SELF.search(text):
        anchors += [(m.start(), 'Self-consistent Calculation')
                    for m in RE_BLOCK_SELF.finditer(text)]
        # 少数输出只在第一步打印该行，其它步从 "iteration #  1" 开始
        anchors += [(m.start(), 'iteration # 1') for m in RE_BLOCK_ITER1.finditer(text)]
    else:
        anchors += [(m.start(), 'iteration # 1') for m in RE_BLOCK_ITER1.finditer(text)]
    anchors.sort()

    # 去重：同一块的两种标志相距很近，二者之间不会出现 "End of self-consistent"
    kept = []
    for pos, label in anchors:
        if not kept:
            kept.append((pos, label))
            continue
        prev = kept[-1][0]
        if pos - prev < 4000 and not RE_END_SCF.search(text[prev:pos]):
            continue          # 同一个块内的第二种标志
        kept.append((pos, label))

    blocks = []
    for k, (pos, label) in enumerate(kept):
        end = kept[k + 1][0] if k + 1 < len(kept) else len(text)
        blocks.append({'index': k + 1, 'label': label,
                       'start': pos, 'end': end, 'text': text[pos:end]})
    return blocks


# --------------------------------------------------------------------------- #
# 力解析（必修 3）
# --------------------------------------------------------------------------- #
RE_FORCE_HEADER = re.compile(
    r'^[ \t]*Forces acting on atoms[ \t]*(?:\([^)]*\))?[ \t]*:[ \t]*$')
RE_FORCE_ATOM = re.compile(
    r'^[ \t]*atom\s+(\d+)\s+type\s+(\d+)\s+force\s*=\s*(' + _F + r')\s+('
    + _F + r')\s+(' + _F + r')\s*$')
RE_FORCE_SUBTITLE = re.compile(r'^[ \t]*The\s+.*\bto forces[ \t]*$')
RE_TOTAL_FORCE = re.compile(
    r'^[ \t]*Total force\s*=\s*(' + _F + r')'
    r'(?:[ \t]+Total SCF correction\s*=\s*(' + _F + r'))?')

# 7 个子表的规范顺序（用于自检：第 1 个必须是总力）
FORCE_SUBTABLE_ORDER = ('total', 'non-local', 'ionic', 'local',
                        'core correction', 'hubbard', 'scf correction')


def parse_force_sections(block_text):
    """解析块内所有 "Forces acting on atoms" 段（通常恰好 1 段）。

    返回 list，每项:
      {'subtables': [{'title':..., 'atoms':[{'i','type','f'(x,y,z)}]}, ...],
       'total_force_rybohr': float|None,
       'total_scf_correction_rybohr': float|None,
       'line': 段标题所在行号}
    第 1 个 subtable 才是总力；其后是 6 个分解子表（必须忽略）。
    """
    out = []
    lines = block_text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        if not RE_FORCE_HEADER.match(lines[i]):
            i += 1
            continue
        sec = {'subtables': [], 'total_force_rybohr': None,
               'total_scf_correction_rybohr': None, 'line': i + 1}
        i += 1
        cur = {'title': 'total', 'atoms': []}
        while i < n:
            ln = lines[i]
            m = RE_FORCE_ATOM.match(ln)
            if m:
                cur['atoms'].append({'i': int(m.group(1)), 'type': int(m.group(2)),
                                     'f': (_f(m.group(3)), _f(m.group(4)), _f(m.group(5)))})
                i += 1
                continue
            if not ln.strip():
                i += 1
                continue
            if RE_FORCE_SUBTITLE.match(ln):
                if cur['atoms']:
                    sec['subtables'].append(cur)
                cur = {'title': ln.strip(), 'atoms': []}
                i += 1
                continue
            mt = RE_TOTAL_FORCE.match(ln)
            if mt:
                sec['total_force_rybohr'] = _f(mt.group(1))
                if mt.group(2) is not None:
                    sec['total_scf_correction_rybohr'] = _f(mt.group(2))
                i += 1
                break
            # 其它非空行（End of ..., Computing stress ...）：力表到此结束
            break
        if cur['atoms']:
            sec['subtables'].append(cur)
        out.append(sec)
    return out


def max_force(forces):
    """max_force = max_i ||F_i||（Ry/bohr 数值，调用方自行换算）。

    forces: 序列，每项 (fx,fy,fz) 或 {'f': (fx,fy,fz)} 或 {'fx':..,'fy':..,'fz':..}。
    """
    vecs = []
    for f in forces or []:
        if isinstance(f, dict):
            if 'f' in f:
                vecs.append(tuple(f['f']))
            else:
                vecs.append((f['fx'], f['fy'], f['fz']))
        else:
            vecs.append(tuple(f))
    if not vecs:
        return None
    return max(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in vecs)


def force_stats(atoms):
    """对一个力子表（atom 列表）给全部统计量。

    返回 dict（能量单位 Ry/bohr 与 eV/Å 两套同时给出）：
      nat, max_atom_norm_rybohr / _eVA    最大原子力矢量模 ||F_i||
      max_atom_index                       该原子序号（1-based）
      max_component_rybohr / _eVA          最大单分量 |F_i,alpha|
      max_component_info                   (原子序号, 'x|y|z')
      rms_force_rybohr / _eVA              sqrt(mean |F_i|^2)
      sum_force_rybohr / _eVA              sum_i F_i（整体平移力，应≈0）
    """
    if not atoms:
        return {}
    vecs = [a['f'] for a in atoms]
    norms = [math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in vecs]
    k = max(range(len(vecs)), key=lambda j: norms[j])
    comp = max(((abs(vecs[j][c]), atoms[j]['i'], 'xyz'[c])
                for j in range(len(vecs)) for c in range(3)), key=lambda t: t[0])
    s = [sum(v[c] for v in vecs) for c in range(3)]
    d = {
        'nat': len(vecs),
        'max_atom_norm_rybohr': norms[k],
        'max_atom_norm_eVA': norms[k] * RY_BOHR_TO_EV_A,
        'max_atom_index': atoms[k]['i'],
        'max_component_rybohr': comp[0],
        'max_component_eVA': comp[0] * RY_BOHR_TO_EV_A,
        'max_component_info': (comp[1], comp[2]),
        'rms_force_rybohr': math.sqrt(sum(x * x for x in norms) / len(norms)),
        'rms_force_eVA': math.sqrt(sum(x * x for x in norms) / len(norms)) * RY_BOHR_TO_EV_A,
        'sum_force_rybohr': tuple(s),
        'sum_force_eVA': tuple(x * RY_BOHR_TO_EV_A for x in s),
    }
    return d


def max_force_from_text(text):
    """便捷函数：全文最后一个力表的总力子表 -> max_i ||F_i||（Ry/bohr）。"""
    secs = parse_force_sections(text)
    if not secs:
        return None
    st = secs[-1]['subtables']
    return max_force(st[0]['atoms']) if st else None


# --------------------------------------------------------------------------- #
# 单块解析
# --------------------------------------------------------------------------- #
RE_FINAL_E = re.compile(r'^!\s+total energy\s*=\s*(' + _F + r')\s*Ry', re.M)
RE_MINUS_TS = re.compile(r'smearing contrib\.\s*\(-TS\)\s*=\s*(' + _F + r')\s*Ry')
RE_E_INTERNAL = re.compile(r'internal energy E=F\+TS\s*=\s*(' + _F + r')\s*Ry')
RE_MAG = re.compile(r'total magnetization\s*=\s*(' + _F + r')\s*Bohr')
RE_ABSMAG = re.compile(r'absolute magnetization\s*=\s*(' + _F + r')\s*Bohr')
RE_FERMI = re.compile(r'the Fermi energy is\s*(' + _F + r')\s*ev')
RE_CONV_OK = re.compile(r'convergence has been achieved in\s+(\d+)\s+iterations')
RE_CONV_BAD = re.compile(r'convergence NOT achieved after\s+(\d+)\s+iterations')
RE_ITER = re.compile(r'^[ \t]*iteration #\s*(\d+)', re.M)
RE_ACC = re.compile(r'estimated scf accuracy\s*<\s*(' + _F + r')')
RE_MOM_SITE = re.compile(
    r'Magnetic moment per site(.*?)(?:\n\s*\n|$)', re.S)
RE_MOM_ROW = re.compile(
    r'atom\s+(\d+)\s+\(R=\s*[^)]*\)\s+charge=\s*(' + _F + r')\s+magn=\s*(' + _F + r')')


def parse_site_moments(text):
    """'Magnetic moment per site' 表 -> {atom_index: magn(μB)}。

    只取文件中**最后一个**该表；块内调用时传块文本，避免跨块混用。
    """
    idx = text.rfind('Magnetic moment per site')
    mom = {}
    if idx < 0:
        return mom
    for ln in text[idx:].splitlines():
        m = RE_MOM_ROW.match(ln.strip()) or RE_MOM_ROW.search(ln)
        if m:
            mom[int(m.group(1))] = _f(m.group(3))
    return mom


def parse_block(block_text, nat=None):
    """解析一个计算块，返回该块的 F / -TS / E / 磁矩 / 力 / 收敛状态。

    返回 dict（键见下），全部为本块内数据，不跨块：
      has_end_scf, n_iter_last, n_iter_lines, last_accuracy_ry
      F_ry, minus_TS_ry, ts_ry, E_internal_ry, E_internal_source
      M_muB, absM_muB, E_fermi_eV
      conv_ok(None/True/False), conv_niter, conv_line, scf_status,
      scf_converged(bool), has_final_energy(bool)
      forces{...force_stats}, force_subtables(int), total_force_rybohr,
      total_scf_correction_rybohr, total_force_eVA
      mom{moments}
    """
    t = block_text
    d = {}
    d['has_end_scf'] = bool(RE_END_SCF.search(t))

    its = [int(m.group(1)) for m in RE_ITER.finditer(t)]
    d['n_iter_lines'] = len(its)
    d['n_iter_last'] = its[-1] if its else None
    accs = [_f(m.group(1)) for m in RE_ACC.finditer(t)]
    d['last_accuracy_ry'] = accs[-1] if accs else None

    # --- 能量（只认带 "!" 的最终行）---
    els = [_f(m.group(1)) for m in RE_FINAL_E.finditer(t)]
    d['F_ry'] = els[-1] if els else None
    d['has_final_energy'] = d['F_ry'] is not None

    # --- 展宽 -TS / E=F+TS ---
    mts = [_f(m.group(1)) for m in RE_MINUS_TS.finditer(t)]
    d['minus_TS_ry'] = mts[-1] if mts else None
    d['ts_ry'] = (-mts[-1]) if mts else None
    eis = [_f(m.group(1)) for m in RE_E_INTERNAL.finditer(t)]
    if eis:
        d['E_internal_ry'] = eis[-1]
        d['E_internal_source'] = 'printed'
    elif d['F_ry'] is not None and d['minus_TS_ry'] is not None:
        # 同块同几何自洽推导，允许但显式标注来源
        d['E_internal_ry'] = d['F_ry'] + d['ts_ry']
        d['E_internal_source'] = 'derived_F_plus_TS'
    else:
        d['E_internal_ry'] = None
        d['E_internal_source'] = None

    mags = [_f(m.group(1)) for m in RE_MAG.finditer(t)]
    absmags = [_f(m.group(1)) for m in RE_ABSMAG.finditer(t)]
    efs = [_f(m.group(1)) for m in RE_FERMI.finditer(t)]
    d['M_muB'] = mags[-1] if mags else None
    d['absM_muB'] = absmags[-1] if absmags else None
    d['E_fermi_eV'] = efs[-1] if efs else None

    # --- 收敛状态（只认本块）---
    ok = list(RE_CONV_OK.finditer(t))
    bad = list(RE_CONV_BAD.finditer(t))
    d['conv_ok'] = True if ok else (False if bad else None)
    if ok:
        d['conv_niter'] = int(ok[-1].group(1))
        d['conv_line'] = ok[-1].group(0).strip()
    elif bad:
        d['conv_niter'] = int(bad[-1].group(1))
        d['conv_line'] = bad[-1].group(0).strip()
    else:
        d['conv_niter'] = None
        d['conv_line'] = None
    if d['conv_ok'] is True:
        d['scf_status'] = 'CONVERGED'
    elif d['conv_ok'] is False:
        d['scf_status'] = 'NOT_CONVERGED'
    elif d['has_final_energy']:
        d['scf_status'] = 'NO_CONVERGENCE_STATEMENT'
    else:
        d['scf_status'] = 'INCOMPLETE'
    # 只有明确打印了 achieved 才算收敛；缺失语句一律不算
    d['scf_converged'] = (d['conv_ok'] is True)

    # --- 力（只取第一个子表）---
    secs = parse_force_sections(t)
    fstats = {}
    nsub = 0
    tf = tscf = None
    if secs:
        sec = secs[-1]
        nsub = len(sec['subtables'])
        tf = sec['total_force_rybohr']
        tscf = sec['total_scf_correction_rybohr']
        if sec['subtables']:
            fstats = force_stats(sec['subtables'][0]['atoms'])
            if nat and fstats and fstats['nat'] != nat:
                fstats['nat_mismatch'] = (fstats['nat'], nat)
    d['forces'] = fstats
    d['force_subtables'] = nsub
    d['first_subtable_title'] = (sec['subtables'][0]['title']
                                 if secs and sec['subtables'] else None)
    # [FIX 2026-10-28c] 力是否完整必须单独成字段（专家 2026-10-28 复审）：
    #   scf_converged 只说明电子收敛，不代表力表完整/序号连续。
    #   forces_complete=True 仅在「有总力子表 + 原子数 == nat + 序号恰为 1..nat」时成立。
    #   纯能量单点可以没有力表（False 不算错误）；几何优化必须与 scf_converged 一起检查。
    d['forces_present'] = bool(secs and sec['subtables'])
    d['forces_complete'] = False
    d['forces_atom_count'] = None
    d['forces_missing_atoms'] = None
    if d['forces_present']:
        atoms0 = sec['subtables'][0]['atoms']
        idx = sorted(a['i'] for a in atoms0)
        nat_expect = nat if nat else (fstats.get('nat') if fstats else None)
        d['forces_atom_count'] = len(atoms0)
        if nat_expect:
            missing = [i for i in range(1, int(nat_expect) + 1) if i not in set(idx)]
            d['forces_missing_atoms'] = missing
            d['forces_complete'] = (len(atoms0) == int(nat_expect) and not missing
                                    and idx == list(range(1, int(nat_expect) + 1)))
    d['total_force_rybohr'] = tf
    d['total_force_eVA'] = (tf * RY_BOHR_TO_EV_A) if tf is not None else None
    d['total_scf_correction_rybohr'] = tscf
    d['mom'] = parse_site_moments(t)

    # --- 自洽性自检：打印的 E 应等于 F+TS ---
    if d['E_internal_source'] == 'printed' and d['F_ry'] is not None and d['ts_ry'] is not None:
        if abs(d['E_internal_ry'] - (d['F_ry'] + d['ts_ry'])) > 1e-6:
            d['E_identity_warn'] = (
                'E_internal(%.8f) != F+TS(%.8f)' % (d['E_internal_ry'], d['F_ry'] + d['ts_ry']))
    return d


# --------------------------------------------------------------------------- #
# 文件级设置信息（口径校验用，必修 4）
# --------------------------------------------------------------------------- #
RE_ECUTWFC = re.compile(r'kinetic-energy cutoff\s*=\s*(' + _F + r')\s*Ry')
RE_ECUTRHO = re.compile(r'charge density cutoff\s*=\s*(' + _F + r')\s*Ry')
RE_KCOORD = re.compile(r'k\(\s*\d+\)\s*=\s*\(\s*(' + _F + r')\s+(' + _F + r')\s+('
                       + _F + r')\s*\),\s*wk =\s*(' + _F + r')')
RE_U = re.compile(r'U\(([A-Za-z0-9_]+)-([a-z0-9]+)\)\s*=\s*(' + _F + r')')
RE_PSEUDO = re.compile(
    r'PseudoPot\.\s*#\s*(\d+)\s+for\s+(\S+)\s+read from file:\s*\n\s*(\S+)')
RE_PSEUDO_ALT = re.compile(r'^\s*file\s+(\S+\.UPF):\s+wavefunction', re.M)
RE_NAT = re.compile(r'number of atoms/cell\s*=\s*(\d+)')
RE_NTYP = re.compile(r'number of atomic types\s*=\s*(\d+)')
RE_ALAT = re.compile(r'lattice parameter \(alat\)\s*=\s*(' + _F + r')\s*a\.u\.')
RE_CELLDM1 = re.compile(r'celldm\(1\)=\s*(' + _F + r')')
RE_TAU = re.compile(r'^\s*\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(\s*(' + _F
                    + r')\s+(' + _F + r')\s+(' + _F + r')\s*\)', re.M)
RE_XC = re.compile(r'Exchange-correlation=\s*([A-Z ]+?)\n')
RE_SCF_THR = re.compile(r'scf convergence threshold\s*=\s*(' + _F + r')')
RE_MIXBETA = re.compile(r'mixing beta\s*=\s*(' + _F + r')')
RE_PROGRAM = re.compile(r'Program\s+(\S+)\s+v\.\s*([\w.]+)')
RE_WALL = re.compile(r'PWSCF\s*:\s*(' + _F + r')s CPU\s+(' + _F + r')s WALL')

# [FIX 2026-10-28b] k 点坐标必须连同单位一起解析。QE 在头部先后打印
#   "cart. coord. in units 2pi/alat" 与 "cryst. coord." 两段；
# 旧实现把两段混在一起去重、又不记录单位，导致
#   (a) nk 相同但坐标不同判不出来（误通过）；
#   (b) 非立方胞里 cart./cryst. 数值不同时会张冠李戴。
RE_KSEC_CART = re.compile(r'^[ \t]*cart\. coord\. in units 2pi/alat[ \t]*$', re.M)
RE_KSEC_CRYST = re.compile(r'^[ \t]*cryst\. coord\.[ \t]*$', re.M)


def _krows_after(text, end):
    """从段头之后收集连续的 k(...) 行；遇到其它行（含空行）即停止。

    段头与第一行之间允许有空行，但表格内部不允许。
    """
    rows = []
    started = False
    for ln in text[end:].splitlines():
        s = ln.strip()
        m = RE_KCOORD.match(s)
        if m:
            started = True
            rows.append((_f(m.group(1)), _f(m.group(2)), _f(m.group(3)), _f(m.group(4))))
            continue
        if not s and not started:
            continue
        break
    return rows


def _k_section(text, header_re):
    """返回 (krows, 段头偏移)，取「其后确实跟着 k 行」的最后一个段；没有则 None。"""
    best = None
    for m in header_re.finditer(text):
        rows = _krows_after(text, m.end())
        if rows:
            best = (rows, m.start())
    return best


def parse_kpoints(text, nk=None):
    """解析 QE 头部 k 点表，并记录坐标系单位（[FIX 2026-10-28b]）。

    返回 dict:
      kpoints          [(kx,ky,kz,wk)]，主坐标系（有 cryst. 就用 cryst.）按值排序
      kpoints_units    'cryst.' 或 'cart.'；没有 k 表时为 None
      kpoints_cryst    只来自 cryst. 段的行（没有则 []）
      kpoints_cart     只来自 cart. 段的行（没有则 []）
      kpoints_nk       头部声明的 k 点数
      kpoints_complete 主坐标段行数 == 声明的 nk（截断文件为 False）

    说明：cryst. 是无量纲分数坐标，cart. 的单位是 2pi/alat，二者不能直接比较；
    立方胞里两段数值恰好相同只是特例，不能据此认为单位无关紧要。
    """
    m = _last(r'number of k points=\s*(\d+)', text)
    declared = int(m.group(1)) if m else nk
    cryst = _k_section(text, RE_KSEC_CRYST)
    cart = _k_section(text, RE_KSEC_CART)
    kc = sorted(cryst[0]) if cryst else []
    ka = sorted(cart[0]) if cart else []
    if cryst and cart:
        # QE 的顺序是先 cart. 后 cryst.；取更靠后的那一段作为主坐标
        primary, units = (kc, 'cryst.') if cryst[1] > cart[1] else (ka, 'cart.')
    elif cryst:
        primary, units = kc, 'cryst.'
    elif cart:
        primary, units = ka, 'cart.'
    else:
        primary, units = [], None
    n_expect = declared if declared else (len(primary) or None)
    return dict(kpoints=primary, kpoints_units=units,
                kpoints_cryst=kc, kpoints_cart=ka,
                kpoints_nk=declared,
                kpoints_complete=bool(primary) and (n_expect is None or len(primary) == n_expect))


def _kpoints_equal(a, b, tol=1e-6):
    """两组 k 点（含权重）是否等价：按值排序后逐分量比较，容差 tol。"""
    a, b = list(a or []), list(b or [])
    if len(a) != len(b):
        return False
    for x, y in zip(sorted(a), sorted(b)):
        if len(x) != len(y) or any(abs(float(u) - float(v)) > tol for u, v in zip(x, y)):
            return False
    return True



def setup_signature(text):
    """提取一次计算的「设置签名」，用于能量口径一致性校验。

    返回 dict:
      ecutwfc, ecutrho            Ry
      nk                          k 点数（对称约化后，QE 打印值）
      smearing, degauss           展宽类型与宽度（Ry）
      kpoints                     [(kx,ky,kz,wk)] 按值排序；有 cryst. 段就用 cryst.
      kpoints_units               'cryst.' / 'cart.'（cart. 单位 2pi/alat），无 k 表为 None
      kpoints_cryst/kpoints_cart  两段各自的行（没有则 []），见 parse_kpoints()
      kpoints_nk / kpoints_complete  声明的 k 点数 / k 行是否齐全（截断为 False）
      U                           {'FeA': 4.0, ...}（eV）
      pseudos                     [(type_index, label, upd_basename)]
      nat, ntyp, alat_bohr, functional, scf_thr, mixing_beta
      geom_fingerprint            几何指纹（tau 表 + alat），用于判断是否同一几何
    """
    d = {}
    m = _last(RE_ECUTWFC, text)
    d['ecutwfc'] = _f(m.group(1)) if m else None
    m = _last(RE_ECUTRHO, text)
    d['ecutrho'] = _f(m.group(1)) if m else None

    d['nk'] = None
    d['smearing'] = None
    d['degauss'] = None
    m = _last(r'number of k points=\s*(\d+)\s+([A-Za-z][A-Za-z ,\-]*?)\s*'
              r'smearing,\s*width \(Ry\)=\s*(' + _F + r')', text)
    if m:
        d['nk'] = int(m.group(1))
        d['smearing'] = re.sub(r'\s+', ' ', m.group(2)).strip()
        d['degauss'] = _f(m.group(3))
    else:
        m = _last(r'number of k points=\s*(\d+)', text)
        d['nk'] = int(m.group(1)) if m else None

    # [FIX 2026-10-28b] k 点坐标连同单位一起解析（旧实现混段去重、无单位，导致误通过）
    d.update(parse_kpoints(text, d['nk']))

    d['U'] = {}
    for m in RE_U.finditer(text):
        d['U'][m.group(1)] = _f(m.group(3))

    pseudos = []
    for m in RE_PSEUDO.finditer(text):
        pseudos.append((int(m.group(1)), m.group(2), os.path.basename(m.group(3))))
    if not pseudos:
        pseudos = [(0, '?', os.path.basename(m.group(1))) for m in RE_PSEUDO_ALT.finditer(text)]
    d['pseudos'] = sorted(set(pseudos))

    m = _last(RE_NAT, text)
    d['nat'] = int(m.group(1)) if m else None
    m = _last(RE_NTYP, text)
    d['ntyp'] = int(m.group(1)) if m else None
    m = _last(RE_ALAT, text) or _last(RE_CELLDM1, text)
    d['alat_bohr'] = _f(m.group(1)) if m else None
    m = _last(RE_XC, text)
    d['functional'] = m.group(1).strip() if m else None
    m = _last(RE_SCF_THR, text)
    d['scf_thr'] = _f(m.group(1)) if m else None
    m = _last(RE_MIXBETA, text)
    d['mixing_beta'] = _f(m.group(1)) if m else None

    # 几何指纹：只认 "site n. atom tau(...)"（= 输入几何的回显），不看运行中更新的坐标
    tau = [(m.group(1), _f(m.group(2)), _f(m.group(3)), _f(m.group(4)))
           for m in RE_TAU.finditer(text)]
    if tau:
        h = hashlib.sha1()
        for lab, x, y, z in tau:
            h.update(('%s %.6f %.6f %.6f;' % (lab, x, y, z)).encode())
        h.update(('alat=%s' % d['alat_bohr']).encode())
        d['geom_fingerprint'] = h.hexdigest()[:16]
        d['geom_nat'] = len(tau)
    else:
        d['geom_fingerprint'] = None
        d['geom_nat'] = None
    return d


def setup_from_file(path_or_text, is_text=False):
    """setup_signature 的便捷入口：给路径或文本。"""
    return setup_signature(path_or_text if is_text else read_text(path_or_text))


# 默认参与硬校验的键（几何对 NEB 各 image 本来就不同，默认不校验）
# [FIX 2026-10-28b] k 点坐标/权重纳入默认校验：nk 相同不代表 k 网格相同。
SETUP_KEYS_DEFAULT = ('ecutwfc', 'ecutrho', 'nk', 'kpoints', 'smearing', 'degauss',
                      'U', 'pseudos')
SETUP_TOL = {'ecutwfc': 1e-6, 'ecutrho': 1e-6, 'degauss': 1e-8}

_SETUP_CN = {
    'ecutwfc': '波函数截断能 ecutwfc (Ry)',
    'ecutrho': '电荷密度截断能 ecutrho (Ry)',
    'nk': 'k 点数 nk',
    'smearing': '展宽方式 smearing',
    'degauss': '展宽宽度 degauss (Ry)',
    'U': 'DFT+U 的 U 值 (eV)',
    'pseudos': '赝势文件',
    'kpoints': 'k 点坐标/权重（默认 cryst. 分数坐标）',
    'kpoints_units': 'k 点坐标系单位（cryst. / cart. = 2pi/alat）',
    'scf_thr': 'SCF 收敛阈值 conv_thr (Ry)',
    'mixing_beta': '混合参数 mixing_beta',
    'functional': '交换关联泛函',
    'nat': '原子数 nat',
    'geom': '几何（tau 表指纹）',
}


def compare_setups(sig_a, sig_b, name_a='A', name_b='B', keys=None,
                   atol=None, raise_error=True):
    """比较两组设置签名（必修 4），返回 {'ok':bool, 'mismatches':[...], 'checked':[...]}。

    不一致时（raise_error=True）抛 SetupMismatchError，消息里写清「哪一项、A 是多少、
    B 是多少、为什么不能直接相减」。
    """
    keys = list(keys if keys is not None else SETUP_KEYS_DEFAULT)
    mismatches = []
    for k in keys:
        if k == 'geom':
            a, b = sig_a.get('geom_fingerprint'), sig_b.get('geom_fingerprint')
            # [FIX 2026-10-28c] 两侧都缺几何信息时旧实现直接 continue → 误判「同一几何」。
            # 缺指纹只能是「无法判断」，不能当成通过（专家 2026-10-28 复审实测 compare_geometry({}, {})）。
            if not a or not b:
                mismatches.append({'key': k, 'a': a, 'b': b,
                                   'why': '至少一侧缺少几何指纹（tau 表缺失或输出被截断），'
                                          '无法判断是否同一几何'})
            elif a != b:
                mismatches.append({'key': k, 'a': a, 'b': b,
                                   'why': '两次计算不是同一几何（几何指纹不同）'})
            continue
        if k == 'kpoints':
            # [FIX 2026-10-28b] k 点必须比「坐标 + 权重」，且只在同一坐标系下比。
            # 旧实现只比 nk 数量（nk 相同但坐标不同会误通过），也不区分 cryst./cart.。
            ka, kb = list(sig_a.get('kpoints') or []), list(sig_b.get('kpoints') or [])
            ua, ub = sig_a.get('kpoints_units'), sig_b.get('kpoints_units')
            if not ka and not kb:
                mismatches.append({'key': k, 'a': ka, 'b': kb,
                                   'why': '两侧都没有 k 点坐标表，无法证明 k 网格一致'})
            elif not ka or not kb:
                mismatches.append({'key': k, 'a': ka, 'b': kb,
                                   'why': '一侧缺少 k 点坐标表（可能被截断），无法比较'})
            elif ua != ub:
                mismatches.append({'key': k, 'a': ka, 'b': kb, 'units': (ua, ub),
                                   'why': 'k 点坐标单位不同（%s vs %s），无法直接比较；'
                                          '需先换算到同一单位（cart. 的单位是 2pi/alat）后重比'
                                          % (ua, ub)})
            elif not _kpoints_equal(ka, kb):
                mismatches.append({'key': k, 'a': ka, 'b': kb,
                                   'why': 'k 点坐标/权重不一致（nk 相同也可能不是同一套 k 网格）'})
            continue
        a, b = sig_a.get(k), sig_b.get(k)
        tol = (atol or {}).get(k, SETUP_TOL.get(k, 0.0)) if isinstance(atol, dict) else (
            SETUP_TOL.get(k, 0.0) if atol is None else atol)
        if k in ('U', 'pseudos'):
            if a != b:
                mismatches.append({'key': k, 'a': a, 'b': b,
                                   'why': '%s 不一致' % _SETUP_CN.get(k, k)})
        elif isinstance(a, float) or isinstance(b, float):
            if a is None or b is None:
                if a != b:
                    mismatches.append({'key': k, 'a': a, 'b': b,
                                       'why': '%s 在一侧缺失（无法证明一致）' % _SETUP_CN.get(k, k)})
            elif abs(float(a) - float(b)) > tol:
                mismatches.append({'key': k, 'a': a, 'b': b,
                                   'why': '%s 不一致（差 %.6g）' % (_SETUP_CN.get(k, k), float(a) - float(b))})
        else:
            if a != b:
                mismatches.append({'key': k, 'a': a, 'b': b,
                                   'why': '%s 不一致' % _SETUP_CN.get(k, k)})
    ok = not mismatches
    rep = {'ok': ok, 'checked': keys, 'mismatches': mismatches}
    if not ok and raise_error:
        msg = ['两次能量的计算设置不一致，不能直接相减（%s vs %s）：' % (name_a, name_b)]
        for mm in mismatches:
            msg.append('  - %s：%s=%r，%s=%r；%s' % (
                mm['why'], name_a, mm['a'], name_b, mm['b'],
                '请统一设置后重算，或明确标注该差值为跨设置外推值'))
        raise SetupMismatchError('\n'.join(msg))
    return rep


def compare_geometry(sig_a, sig_b, name_a='A', name_b='B', raise_error=True):
    """只校验几何是否同一（F 与 -TS 必须来自同一几何，必修 5）。

    注意：任一侧缺几何指纹（输出被截断 / 无 tau 表）时返回 ok=False，
    即「无法判断」，不会静默算作同一几何。
    """
    return compare_setups(sig_a, sig_b, name_a, name_b, keys=('geom',),
                          raise_error=raise_error)


def check_energy_convention(edef_a, edef_b, ts_a=None, ts_b=None,
                            name_a='A', name_b='B', tol=1e-8, raise_error=True):
    """校验两次能量的「能量定义」是否一致（必修 4）。

    edef_*: 'F'  表示自由能 F（含 -TS，QE 的 "! total energy"）
            'E'  表示内能 E = F + TS（QE 的 "internal energy E=F+TS"）
    ts_a/ts_b: 对应 -TS 值（Ry），用于确认 E 口径确实带了展宽修正。
    返回 {'ok':bool, 'why':str}；不一致抛 SetupMismatchError。
    """
    probs = []
    if edef_a != edef_b:
        probs.append('能量定义不同：%s 用 %s，%s 用 %s（%s）' % (
            name_a, edef_a, name_b, edef_b,
            'F 与 E=F+TS 相差 -TS，金属体系可达 0.3 Ry 量级，严禁混用'))
    if edef_a == 'E' and ts_a is None:
        probs.append('%s 声称 E 口径但没有 -TS 记录，无法证明展宽已计入' % name_a)
    if edef_b == 'E' and ts_b is None:
        probs.append('%s 声称 E 口径但没有 -TS 记录，无法证明展宽已计入' % name_b)
    if edef_a == 'F' and edef_b == 'F':
        # F 口径本身不需要 TS；但若两侧 TS 差很大，提醒改用 E 口径比较
        if ts_a is not None and ts_b is not None and abs(ts_a - ts_b) > 1e-3:
            probs.append('两侧都用 F 口径，但 -TS 相差 %.4f Ry（约 %.1f eV）：'
                         'F 口径的差值包含展宽熵项，建议改用 E 口径' % (
                             ts_a - ts_b, (ts_a - ts_b) * RY_TO_EV))
    ok = not probs
    if not ok and raise_error:
        raise SetupMismatchError('能量口径不一致：\n  - ' + '\n  - '.join(probs))
    return {'ok': ok, 'problems': probs}


def check_energy_identity(ea_fwd, ea_bwd, e_final, e_initial, tol=1e-6,
                          label='', raise_error=True):
    """恒等式校验（必修 4）：E_a(→) - E_a(←) = E_final - E_initial，单位 eV。

    推导：E_a(→) = E_TS - E_initial，E_a(←) = E_TS - E_final
          => E_a(→) - E_a(←) = E_final - E_initial。
    返回 {'ok','lhs','rhs','diff','tol','label'}；超出容差抛 QEParseError。
    """
    lhs = ea_fwd - ea_bwd
    rhs = e_final - e_initial
    diff = lhs - rhs
    ok = abs(diff) <= tol
    rep = {'ok': ok, 'lhs': lhs, 'rhs': rhs, 'diff': diff, 'tol': tol, 'label': label}
    if not ok and raise_error:
        raise QEParseError(
            '%s恒等式校验失败：Ea(→)-Ea(←)=%.9f eV，E_final-E_initial=%.9f eV，'
            '差 %.3e eV > 容差 %.0e eV（能量/图像/端点数据互相矛盾）'
            % ((label + ' ') if label else '', lhs, rhs, diff, tol))
    return rep


def check_barrier_table(energies_eV, ea_fwd, ea_bwd, tol=1e-6, label='', raise_error=True):
    """校验 image 表与打印的 activation energy 自洽（并做恒等式校验）。

    energies_eV: 按 image 顺序的总能量（eV）；ea_fwd/ea_bwd: neb.x 打印值（eV）。
    检查 max(E) 对应图像、Ea(→)=max-E0、Ea(←)=max-EN、以及恒等式。
    """
    rep = {'ok': True, 'checks': [], 'imax': None}
    e0, eN = energies_eV[0], energies_eV[-1]
    imax = max(range(len(energies_eV)), key=lambda i: energies_eV[i])
    rep['imax'] = imax + 1
    for name, got, want in (('Ea(->)', ea_fwd, energies_eV[imax] - e0),
                            ('Ea(<-)', ea_bwd, energies_eV[imax] - eN)):
        d = got - want
        ok = abs(d) <= tol
        rep['checks'].append({'name': name, 'printed': got, 'from_table': want, 'diff': d, 'ok': ok})
        if not ok:
            rep['ok'] = False
    rep['identity'] = check_energy_identity(ea_fwd, ea_bwd, eN, e0, tol=tol,
                                            label=label, raise_error=False)
    rep['ok'] = rep['ok'] and rep['identity']['ok']
    if not rep['ok'] and raise_error:
        bad = [c for c in rep['checks'] if not c['ok']]
        msg = ['%s势垒表自洽性校验失败：' % ((label + ' ') if label else '')]
        for c in bad:
            msg.append('  %s 打印 %.6f eV，而由 image 表算得 %.6f eV（差 %.2e）'
                       % (c['name'], c['printed'], c['from_table'], c['diff']))
        if not rep['identity']['ok']:
            msg.append('  恒等式 Ea(→)-Ea(←)=E_final-E_initial 不成立（差 %.2e eV）'
                       % rep['identity']['diff'])
        raise QEParseError('\n'.join(msg))
    return rep


def energy_of(block, edef='E'):
    """按指定能量定义取一个块的能量（Ry）。'E' -> E=F+TS，'F' -> F。

    E 口径下若块内没有 -TS/内能，抛 QEParseError（禁止把 F 当 E）。
    """
    if edef == 'F':
        if block.get('F_ry') is None:
            raise QEParseError('该块没有 F（! total energy 行）')
        return block['F_ry']
    if edef == 'E':
        if block.get('E_internal_ry') is None:
            raise QEParseError(
                '该块没有 E=F+TS（无 internal energy 行，也无 -TS），'
                '禁止用 F 冒充 E 口径')
        return block['E_internal_ry']
    raise QEParseError('未知能量定义 %r（只支持 F 或 E）' % (edef,))


# --------------------------------------------------------------------------- #
# 文件级入口：pw.x
# --------------------------------------------------------------------------- #
def parse_relax_info(text):
    """relax / vc-relax 的离子步与终止原因（与 SCF 收敛分开记录）。"""
    d = {}
    d['bfgs_steps'] = len(re.findall(r'number of bfgs steps', text))
    d['bfgs_converged'] = ('bfgs converged' in text) or (
        'End of BFGS Geometry Optimization' in text)
    d['maxsteps_reached'] = 'reached the maximum number of steps' in text
    d['curvature_fail'] = len(re.findall(r'bfgs curvature condition failed', text))
    d['scf_notconv_events'] = len(re.findall(r'convergence NOT achieved', text))
    d['job_done'] = 'JOB DONE' in text
    m = _last(r'Final energy\s*=\s*(' + _F + r')\s*Ry', text)
    d['final_energy_ry'] = _f(m.group(1)) if m else None
    m = _last(r'Total force\s*=\s*(' + _F + r')', text)
    d['final_total_force_rybohr'] = _f(m.group(1)) if m else None
    lines = [ln.strip() for ln in text.splitlines()
             if re.search(r'converged|convergence|maximum number of steps|'
                          r'End of BFGS|bfgs steps', ln, re.I)]
    d['status_lines'] = lines[-8:]
    return d


def parse_qe_out(path=None, text=None, strict=False, nat=None, edef='E'):
    """解析一个 pw.x 输出文件（以最后一个计算块为准）。

    参数
    ----
    path / text : 二选一（text 便于单元测试）
    strict      : True 时若最后一个块未收敛，抛 QEConvergenceError（harvest 脚本用）
    nat         : 原子数（默认从文件里读）
    edef        : 期望的能量定义 'E' 或 'F'；用于标注，不做静默替换

    返回
    ----
    dict，包含（节选）：
      file, size, mtime, program, qe_version, job_done
      n_blocks, block_index, block_label, block_text_head
      F_ry, minus_TS_ry, ts_ry, E_internal_ry, E_internal_source,
      M_muB, absM_muB, E_fermi_eV,
      scf_converged, scf_status, scf_niter, conv_line,
      forces_present, forces_complete, forces_atom_count, forces_missing_atoms
        （forces_complete 与 scf_converged 相互独立：几何优化必须同时检查两者）
      max_atom_norm_rybohr/_eVA, max_atom_index, max_component_rybohr/_eVA,
      max_component_info, rms_force_rybohr/_eVA, total_force_rybohr/_eVA,
      total_scf_correction_rybohr, force_subtables, forces, mom, types,
      ecutwfc, ecutrho, nk, smearing, degauss, U, pseudos, geom_fingerprint...
      setup: setup_signature() 的完整 dict
      relax: parse_relax_info() 的结果
      warnings: [中文提示]
    以及旧字段别名（向后兼容）：F, E_internal, M, absM, E_F, nscf_iter,
      maxF_rybohr, maxF_eVA, maxF_atom, total_force, cpu_s, wall_s。
    """
    if text is None:
        if path is None:
            raise ValueError('需要 path 或 text')
        text = read_text(path)
    d = {'file': path, 'size': (os.path.getsize(path) if path and os.path.exists(path) else
                                len(text.encode('utf-8'))),
         'mtime': (time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(os.path.getmtime(path)))
                   if path and os.path.exists(path) else None),
         'warnings': []}
    m = RE_PROGRAM.search(text)
    d['program'] = m.group(1) if m else None
    d['qe_version'] = m.group(2) if m else None
    d['job_done'] = 'JOB DONE' in text
    m = _last(RE_WALL, text)
    d['cpu_s'] = _f(m.group(1)) if m else None
    d['wall_s'] = _f(m.group(2)) if m else None

    setup = setup_signature(text)
    d['setup'] = setup
    for k in ('ecutwfc', 'ecutrho', 'nk', 'smearing', 'degauss', 'U', 'pseudos',
              'nat', 'ntyp', 'alat_bohr', 'functional', 'scf_thr', 'mixing_beta',
              'geom_fingerprint', 'kpoints'):
        d[k] = setup.get(k)
    if nat is None:
        nat = setup.get('nat')
    d['nat'] = nat

    blocks = split_calc_blocks(text)
    d['n_blocks'] = len(blocks)
    d['block_index'] = blocks[-1]['index'] if blocks else None
    d['block_label'] = blocks[-1]['label'] if blocks else None
    d['scf_blocks_converged'] = sum(
        1 for b in blocks if parse_block(b['text'], nat=nat)['scf_converged'])

    blk = parse_block(blocks[-1]['text'], nat=nat) if blocks else parse_block('', nat=nat)
    if not blocks:
        d['warnings'].append('文件里没有任何计算块（可能被截断在 SCF 之前）：%s'
                             % (path or '<text>'))
    d['block'] = blk
    d['F_ry'] = blk['F_ry']
    d['minus_TS_ry'] = blk['minus_TS_ry']
    d['ts_ry'] = blk['ts_ry']
    d['E_internal_ry'] = blk['E_internal_ry']
    d['E_internal_source'] = blk['E_internal_source']
    d['M_muB'] = blk['M_muB']
    d['absM_muB'] = blk['absM_muB']
    d['E_fermi_eV'] = blk['E_fermi_eV']
    d['scf_converged'] = blk['scf_converged']
    d['scf_status'] = blk['scf_status']
    d['scf_niter'] = blk['conv_niter']
    d['conv_line'] = blk['conv_line']
    d['last_accuracy_ry'] = blk['last_accuracy_ry']
    d['mom'] = blk['mom']
    d['forces'] = blk['forces']
    d['force_subtables'] = blk['force_subtables']
    d['forces_present'] = blk.get('forces_present')
    d['forces_complete'] = blk.get('forces_complete')
    d['forces_atom_count'] = blk.get('forces_atom_count')
    d['forces_missing_atoms'] = blk.get('forces_missing_atoms')
    d['total_force_rybohr'] = blk['total_force_rybohr']
    d['total_force_eVA'] = blk['total_force_eVA']
    d['total_scf_correction_rybohr'] = blk['total_scf_correction_rybohr']

    fs = blk['forces'] or {}
    d['max_atom_norm_rybohr'] = fs.get('max_atom_norm_rybohr')
    d['max_atom_norm_eVA'] = fs.get('max_atom_norm_eVA')
    d['max_atom_index'] = fs.get('max_atom_index')
    d['max_component_rybohr'] = fs.get('max_component_rybohr')
    d['max_component_eVA'] = fs.get('max_component_eVA')
    d['max_component_info'] = fs.get('max_component_info')
    d['rms_force_rybohr'] = fs.get('rms_force_rybohr')
    d['rms_force_eVA'] = fs.get('rms_force_eVA')
    d['sum_force_rybohr'] = fs.get('sum_force_rybohr')

    # 旧字段别名（历史上其它脚本用过；语义保持不变）
    d['F'] = d['F_ry']
    d['E_internal'] = d['E_internal_ry']
    d['M'] = d['M_muB']
    d['absM'] = d['absM_muB']
    d['E_F'] = d['E_fermi_eV']
    d['nscf_iter'] = d['scf_niter']
    d['maxF_rybohr'] = d['max_atom_norm_rybohr']
    d['maxF_eVA'] = d['max_atom_norm_eVA']
    d['maxF_atom'] = d['max_atom_index']
    d['total_force'] = d['total_force_rybohr']

    # 原子类型（ATOMIC_POSITIONS 最后一个块；pw.x 输出里可能没有）
    types = []
    pos = list(re.finditer(r'ATOMIC_POSITIONS\s*\([a-z]+\)(.*?)(?:K_POINTS|$)', text, re.S))
    if pos:
        for line in pos[-1].group(1).strip().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    float(parts[1])
                except ValueError:
                    continue
                types.append(parts[0])
    d['types'] = types

    d['relax'] = parse_relax_info(text)

    # ---- 结论性提示 ----
    if blocks and not d['scf_converged']:
        d['warnings'].append('最后一个计算块（第 %d 块）SCF 状态 = %s：该块的 F/力不可作为最终结果'
                             % (d['block_index'], d['scf_status']))
    if d['E_internal_source'] == 'derived_F_plus_TS':
        d['warnings'].append('E=F+TS 由同块 F 与 -TS 推导（QE 未打印 internal energy 行）')
    if d['E_internal_ry'] is None and d['F_ry'] is not None:
        d['warnings'].append('该块没有 -TS / E=F+TS：若 occupations=fixed 则 E 恒等于 F，'
                             '本解析不擅自替代，E 口径留空')
    if blk.get('E_identity_warn'):
        d['warnings'].append('内部自洽性异常：' + blk['E_identity_warn'])
    if d['force_subtables'] and d['force_subtables'] not in (1, 7, 8):
        d['warnings'].append('力表子表数 = %d（正常 1 或 7；带 Martyna-Tuckerman 修正时为 8）'
                             % d['force_subtables'])
    if blk.get('first_subtable_title') not in (None, 'total'):
        d['warnings'].append('力表第一个子表不是总力（标题=%r）：解析可能错位'
                             % blk['first_subtable_title'])
    if d.get('nat') and fs.get('nat_mismatch'):
        d['warnings'].append('力表原子数 %s != nat %s' % fs['nat_mismatch'])
    # [FIX 2026-10-28c] 力完整性告警：SCF 收敛 != 力可作收敛判据
    if d.get('forces_present') and not d.get('forces_complete'):
        d['warnings'].append(
            '力表不完整（%s 行 / nat=%s，缺失原子 %s）：不能用于几何收敛判据，'
            '只能作参考' % (d.get('forces_atom_count'), d.get('nat'),
                            d.get('forces_missing_atoms')))
    if not d.get('forces_present') and d.get('nat'):
        d['warnings'].append('该块没有总力表（纯能量单点可以如此；几何优化不能以此判收敛）')

    # [FIX 2026-10-28c] 相对完整的返回字段说明见函数 docstring；此处只补力完整性结论
    if strict and not blocks:
        raise QEConvergenceError(
            '%s：文件里没有任何计算块（SCF 尚未开始或输出被截断），无法判定收敛' % (
                os.path.basename(path) if path else '<text>'))
    if strict and not d['scf_converged']:
        raise QEConvergenceError(
            '%s：最后一个计算块（第 %d 块，%s）SCF 未收敛（%s）' % (
                os.path.basename(path) if path else '<text>',
                d['block_index'], d['block_label'], d['conv_line'] or '无收敛语句'))
    return d


# --------------------------------------------------------------------------- #
# 文件级入口：neb.x
# --------------------------------------------------------------------------- #
RE_NEB_ITER = re.compile(r'^[ \t]*-{5,}[ \t]*iteration\s+(\d+)[ \t]*-{5,}[ \t]*$', re.M)
RE_NEB_NIMG = re.compile(r'num_of_images\s*=\s*(\d+)')
RE_NEB_NSTEP = re.compile(r'nstep_path\s*=\s*(\d+)')
RE_NEB_CI = re.compile(r'CI_scheme\s*=\s*(\S+)')
RE_NEB_OPTS = re.compile(r'opt_scheme\s*=\s*(\S+)')
RE_NEB_KMODE = re.compile(r'string_method\s*=\s*(\S+)')
RE_NEB_PATHT = re.compile(r'path_thr\s*=\s*(' + _F + r')\s*eV\s*/\s*A', re.I)
RE_NEB_EA_F = re.compile(r'activation energy\s*\(->\)\s*=\s*(' + _F + r')\s*eV')
RE_NEB_EA_B = re.compile(r'activation energy\s*\(<\s*-?\s*\)\s*=\s*(' + _F + r')\s*eV')
RE_NEB_PATHLEN = re.compile(r'path length\s*=\s*(' + _F + r')\s*bohr')
RE_NEB_INTER = re.compile(r'inter-image distance\s*=\s*(' + _F + r')\s*bohr')
RE_NEB_INITPL = re.compile(r'initial path length\s*=\s*(' + _F + r')\s*bohr')
RE_NEB_CIIMG = re.compile(r'climbing image\s*=\s*(\d+)')
RE_NEB_SCFFAIL = re.compile(r'scf convergence NOT achieved on image\s+(\d+)')
RE_NEB_IMG_ROW = re.compile(r'^\s*(\d+)\s+(' + _F + r')\s+(' + _F + r')\s+([TF])\s*$')
RE_NEB_IMG_HDR = re.compile(r'^\s*image\s+energy \(eV\)\s+error \(eV/A\)')


def _neb_iteration(seg, nimgs=None):
    """解析 neb.x 一轮迭代的输出段（段 = 一条 iteration 标题到下一个标题之间）。"""
    it = {'ea_fwd_eV': None, 'ea_bwd_eV': None, 'pathlen_bohr': None,
          'inter_image_bohr': None, 'ci_image': None,
          'energies_eV': [], 'errors_eVA': [], 'frozen': [], 'n_rows': 0,
          'scf_fail_images': [], 'n_scf_fail_events': 0}
    m = _last(RE_NEB_EA_F, seg)
    it['ea_fwd_eV'] = _f(m.group(1)) if m else None
    m = _last(RE_NEB_EA_B, seg)
    it['ea_bwd_eV'] = _f(m.group(1)) if m else None
    m = _last(RE_NEB_PATHLEN, seg)
    it['pathlen_bohr'] = _f(m.group(1)) if m else None
    m = _last(RE_NEB_INTER, seg)
    it['inter_image_bohr'] = _f(m.group(1)) if m else None
    m = RE_NEB_CIIMG.search(seg)
    it['ci_image'] = int(m.group(1)) if m else None
    it['scf_fail_images'] = sorted(set(int(x.group(1)) for x in RE_NEB_SCFFAIL.finditer(seg)))
    it['n_scf_fail_events'] = len(RE_NEB_SCFFAIL.findall(seg))

    lines = seg.splitlines()
    for i, ln in enumerate(lines):
        if not RE_NEB_IMG_HDR.match(ln):
            continue
        rows_e, rows_err, rows_fr = [], [], []
        for ln2 in lines[i + 1:]:
            mm = RE_NEB_IMG_ROW.match(ln2)
            if not mm:
                if not ln2.strip():
                    continue
                break
            rows_e.append(_f(mm.group(2)))
            rows_err.append(_f(mm.group(3)))
            rows_fr.append(mm.group(4))
        # 取该段里行数最多的一张表（正常只有一张）
        if len(rows_e) > len(it['energies_eV']):
            it['energies_eV'], it['errors_eVA'], it['frozen'] = rows_e, rows_err, rows_fr
            it['n_rows'] = len(rows_e)
    # 完整性：image 表行数必须等于 num_of_images，且 Ea/pathlen 都在
    it['complete'] = (bool(nimgs) and it['n_rows'] == nimgs
                      and it['ea_fwd_eV'] is not None
                      and it['ea_bwd_eV'] is not None
                      and it['pathlen_bohr'] is not None)
    return it


def parse_neb_out(path=None, text=None):
    """解析 neb.x 输出（与 pw.x 单点完全不同的文件类型）。

    逐轮解析：neb.x 每轮打印 activation energy / image 表 / path length。
    **最后一轮可能被截断**（作业被杀/未写全），因此不是「取最后出现的字段」，
    而是取「最后一个**完整**轮次」，并显式记录最后一轮不完整。

    返回 dict:
      nimgs, nstep_path, ci_scheme, opt_scheme, string_method, path_thr_eVA,
      initial_path_bohr, n_iter_blocks, iterations[每轮 dict],
      last_iter(最后一轮编号), last_complete_iter(最后一个完整轮编号),
      source_iter(上面 energies/Ea 实际取自哪一轮),
      complete(该轮 image 表是否完整), tail_truncated(最后一轮是否残缺),
      energies_eV / errors_eVA / frozen / n_rows,
      ea_fwd_eV, ea_bwd_eV, pathlen_bohr, inter_image_bohr, ci_image,
      job_done, maxsteps_reached, neb_converged(None/True/False),
      scf_fail_images / scf_all_converged（该轮 image 级 SCF 是否全收敛）,
      status: 'NO_ITERATION' | 'FAILED_SCF' | 'COMPLETE' |
              'TRUNCATED_LAST_ITER'（最后一轮残缺，但有完整轮可用）|
              'NO_COMPLETE_TABLE'
      warnings
    """
    if text is None:
        if path is None:
            raise ValueError('需要 path 或 text')
        text = read_text(path)
    d = {'file': path, 'warnings': []}

    def _one(rx, txt, cast=float):
        m = _last(rx, txt)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except (TypeError, ValueError):
            return None

    d['nimgs'] = _one(RE_NEB_NIMG, text, int)
    d['nstep_path'] = _one(RE_NEB_NSTEP, text, int)
    d['ci_scheme'] = _one(RE_NEB_CI, text, str)
    d['opt_scheme'] = _one(RE_NEB_OPTS, text, str)
    d['string_method'] = _one(RE_NEB_KMODE, text, str)
    d['path_thr_eVA'] = _one(RE_NEB_PATHT, text)
    d['initial_path_bohr'] = _one(RE_NEB_INITPL, text)
    d['job_done'] = 'JOB DONE' in text
    d['maxsteps_reached'] = bool(re.search(r'reached the maximum number of steps', text))

    its = [(int(m.group(1)), m.start()) for m in RE_NEB_ITER.finditer(text)]
    d['n_iter_blocks'] = len(its)
    d['last_iter'] = its[-1][0] if its else None
    d['iterations'] = []
    for k, (num, pos) in enumerate(its):
        end = its[k + 1][1] if k + 1 < len(its) else len(text)
        it = _neb_iteration(text[pos:end], d['nimgs'])
        it['iter'] = num
        d['iterations'].append(it)

    complete_iters = [it for it in d['iterations'] if it['complete']]
    d['last_complete_iter'] = complete_iters[-1]['iter'] if complete_iters else None
    last = d['iterations'][-1] if d['iterations'] else None
    src = complete_iters[-1] if complete_iters else last
    d['tail_truncated'] = bool(last) and not last['complete']
    d['complete'] = bool(last) and last['complete'] and src is last
    d['source_iter'] = src['iter'] if src else None

    if src:
        d['energies_eV'] = src['energies_eV']
        d['errors_eVA'] = src['errors_eVA']
        d['frozen'] = src['frozen']
        d['n_rows'] = src['n_rows']
        d['ea_fwd_eV'] = src['ea_fwd_eV']
        d['ea_bwd_eV'] = src['ea_bwd_eV']
        d['pathlen_bohr'] = src['pathlen_bohr']
        d['inter_image_bohr'] = src['inter_image_bohr']
        d['ci_image'] = src['ci_image']
        d['scf_fail_images'] = src['scf_fail_images']
    else:
        d.update({'energies_eV': [], 'errors_eVA': [], 'frozen': [], 'n_rows': 0,
                  'ea_fwd_eV': None, 'ea_bwd_eV': None, 'pathlen_bohr': None,
                  'inter_image_bohr': None, 'ci_image': None, 'scf_fail_images': []})
    # [FIX 2026-10-28c] 区分两个含义（专家 2026-10-28 复审）：
    #   no_scf_failure_reported = 输出里没打印 "scf convergence NOT achieved on image"
    #   scf_all_converged       = 逐 image 核验都收敛。neb.x 只在失败时打印失败信息，
    #                            成功时仅打印 tcpu，因此从 neb.out 本身无法确认「全部收敛」；
    #                            真值需逐 image 子目录输出。这里不把「没报错」当「已收敛」，
    #                            有明确失败时给 False，无失败时留 None（未核实）。
    d['no_scf_failure_reported'] = bool(src) and not d['scf_fail_images']
    d['scf_all_converged'] = False if d['scf_fail_images'] else None

    conv_lines = [ln.strip() for ln in text.splitlines()
                  if re.search(r'converg', ln, re.I) and 'NOT achieved' not in ln
                  and 'scf convergence' not in ln and 'self-consistency' not in ln
                  and 'scf cycles' not in ln]
    d['convergence_lines'] = conv_lines[-5:]
    d['neb_converged'] = True if any(
        re.search(r'convergence (achieved|reached)|converged after|path threshold',
                  ln, re.I) for ln in conv_lines) else (
        False if d['maxsteps_reached'] else None)

    if not its:
        d['status'] = 'NO_ITERATION'
    elif d['scf_fail_images']:
        d['status'] = 'FAILED_SCF'
    elif d['complete']:
        d['status'] = 'COMPLETE'
    elif d['tail_truncated'] and complete_iters:
        d['status'] = 'TRUNCATED_LAST_ITER'
    else:
        d['status'] = 'NO_COMPLETE_TABLE'

    if d['job_done'] and not d['no_scf_failure_reported']:
        d['warnings'].append('JOB DONE 只代表进程正常结束：该轮仍有 image %s 的 SCF 未收敛'
                             % d['scf_fail_images'])
    if d['tail_truncated'] and complete_iters:
        d['warnings'].append(
            '最后一轮（iteration %s）输出被截断（无完整 image 表）；'
            '以上数值取自最后一个完整轮 iteration %s（Ea/pathlen 若与末轮不同属正常）'
            % (d['last_iter'], d['last_complete_iter']))
    elif not complete_iters and its:
        d['warnings'].append('没有任何一轮输出完整的 image 表（rows<num_of_images 或缺少 '
                             'Ea/path length）：不能据此计算势垒')
    if d['job_done'] and not d['maxsteps_reached'] and not d['complete'] and its:
        d['warnings'].append('JOB DONE 但没有完整末轮：作业可能在写结果前被杀')
    return d


# --------------------------------------------------------------------------- #
# 几何 / 原子表
# --------------------------------------------------------------------------- #
def parse_positions(path, which='last', text=None):
    """从输出里取 ATOMIC_POSITIONS 块 -> [(label, x, y, z)]（原样单位，通常 Å）。"""
    t = text if text is not None else read_text(path)
    blocks = list(re.finditer(
        r'ATOMIC_POSITIONS\s*\(\s*(\w+)\s*\)(.*?)(?=\n\s*\n|\n[A-Z_ ]{3,}\n|$)', t, re.S))
    if not blocks:
        return []
    blk = blocks[-1] if which == 'last' else blocks[0]
    out = []
    for line in blk.group(2).strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                out.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return out


def parse_species_table(path, text=None):
    """'site n. atom tau(...)' 表 -> [(index, label, tau_x, tau_y, tau_z)]（alat 单位）。"""
    t = text if text is not None else read_text(path)
    out = []
    for m in re.finditer(r'^\s*(\d+)\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(\s*(' + _F
                         + r')\s+(' + _F + r')\s+(' + _F + r')\s*\)', t, re.M):
        out.append((int(m.group(1)), m.group(2), _f(m.group(3)), _f(m.group(4)), _f(m.group(5))))
    return out


# --------------------------------------------------------------------------- #
# 汇总输出（沿用旧接口，字段更全）
# --------------------------------------------------------------------------- #
def summarize(d, extra=()):
    """压缩成一行可读结果（供 CSV / 日志）。单位在键名里写明。"""
    keys = ['F_ry', 'ts_ry', 'E_internal_ry', 'M_muB', 'absM_muB', 'E_fermi_eV',
            'nk', 'ecutwfc', 'ecutrho', 'degauss',
            'max_atom_norm_rybohr', 'max_atom_norm_eVA', 'max_atom_index',
            'max_component_eVA', 'rms_force_eVA',
            'total_force_rybohr', 'total_force_eVA',
            'scf_converged', 'scf_status', 'scf_niter', 'job_done',
            'block_index', 'n_blocks']
    out = {k: d.get(k) for k in keys}
    for k in extra:
        out[k] = d.get(k)
    return out


def _cli(argv):
    """命令行：python qe_parse.py OUT [OUT ...]（自动区分 pw.x / neb.x）。"""
    for p in argv:
        if not os.path.exists(p):
            print('MISSING %s' % p)
            continue
        t = read_text(p)
        if re.search(r'^\s*Program\s+NEB\b', t, re.M):
            neb = parse_neb_out(path=p)
            print('== %s (neb.x, %.1f KB)' % (os.path.basename(p), os.path.getsize(p) / 1024.0))
            print(json.dumps({k: v for k, v in neb.items() if k != 'warnings'},
                             indent=1, ensure_ascii=False, default=str))
            for w in neb['warnings']:
                print('  ! ' + w)
        else:
            d = parse_qe_out(path=p)
            print('== %s (pw.x, %.1f KB)' % (os.path.basename(p), os.path.getsize(p) / 1024.0))
            print(json.dumps(summarize(d), indent=1, ensure_ascii=False, default=str))
            for w in d['warnings']:
                print('  ! ' + w)
        print('')


if __name__ == '__main__':
    import sys
    _cli(sys.argv[1:])
