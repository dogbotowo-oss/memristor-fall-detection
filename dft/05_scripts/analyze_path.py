#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_path.py —— 静态线性路径 / NEB 路径的能量剖面与势垒。

输入：按路径顺序排列的 pw.x 输出文件（path0..pathN，或任一串单点/弛豫输出）。
输出（写到 --outdir）：
  path_energies.csv : image,t,E_FplusTS_Ry,TS_Ry,F_Ry,dE_eV,M_muB,scf_status,block_index,job_done,file
  neb.dat           : t  E(ref)-E_ref[eV]  F(ref)-E_ref[eV]  TS_Ry（仅完整路径才写）
  summary.txt       : 势垒、口径校验、缺口报告
单位：Ry（能量）→ eV 用 RY_TO_EV=13.605693122994；力 Ry/bohr → eV/Å 用 25.7110456。
假设：
  1) 每个文件贡献一个 image；路径顺序 = 命令行顺序。t 按等间距 i/(N-1) 归一化
     （静态线性插值就是等间距；若实际不是等间距，t 只是相对位置，不要当反应坐标读数）。
  2) 能量口径为 E = F + TS（internal energy）。若某 image 缺 E（如 occupations=fixed），
     留空并报告，**不会**用 F 顶替；可用 --edef F 显式声明整条路径都用 F 口径。
  3) 只有**全部 image 都在场且各自最后一块 SCF 收敛**时才给出最终势垒；
     缺任何一个就明确报「不完整」，绝不从剩余 image 里挑最高点当最终势垒
     （--allow-incomplete 只输出诊断剖面，并在标题与 summary 中标注）。
  4) 势垒一律由 image 能量表重算（Ea(->)=max(E)-E(first)，Ea(<-)=max(E)-E(last)），
     并校验恒等式 Ea(->) - Ea(<-) = E_final - E_initial（容差 1e-6 eV）以及
     所有 image 的截断能/k 点/展宽/U/赝势/几何指纹是否一致（不一致直接报错）。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qe_parse as q  # noqa: E402

RY_EV = q.RY_TO_EV
RY_BOHR_EV_A = q.RY_BOHR_TO_EV_A


def parse_image(fn, edef='E'):
    """解析单个 image 的 pw.x 输出；返回的 E/F/TS/M/conv 全部来自同一个计算块。"""
    d = q.parse_qe_out(path=fn, edef=edef)
    return dict(file=fn, F=d['F_ry'], TS=d['ts_ry'], E=energy(d, edef),
                M=d['M_muB'], conv=d['scf_converged'], status=d['scf_status'],
                block=d['block_index'], jobdone=bool(d['relax']['job_done']),
                maxF_eVA=d['maxF_eVA'], setup=d['setup'], warnings=d['warnings'])


def energy(d, edef='E'):
    """按声明的口径取能量（Ry）；E 口径缺值返回 None，绝不回退到 F。"""
    if edef == 'F':
        return d['F_ry']
    return d['E_internal_ry']


def analyze(files, edef='E', allow_incomplete=False, nimg=None, log=print):
    """核心逻辑（可测试）：返回 report dict，不依赖 argparse。"""
    # 期望 image 数：--nimg 给了就用它，否则只能相信文件数（此时无法发现「少传了文件」）
    expected = nimg if nimg else len(files)
    if nimg is not None and nimg != len(files):
        log('!! --nimg=%d 与给出的文件数 %d 不符：按 %d 个 image 的路径检查完整性'
            % (nimg, len(files), nimg))
    rows = []
    for i, fn in enumerate(files):
        t = i / float(expected - 1) if expected > 1 else 0.0
        try:
            d = parse_image(fn, edef=edef)
        except Exception as e:
            log('!! image %d 解析失败：%s（%s）' % (i, os.path.basename(fn), e))
            rows.append(dict(file=fn, t=t, F=None, TS=None, E=None, M=None, conv=False,
                             status='UNREADABLE', block=None, jobdone=False, maxF_eVA=None,
                             setup=None, warnings=[str(e)]))
            continue
        d['t'] = t
        rows.append(d)

    # ---- 必修 4：口径一致性（只比数值设置） ----
    # 注意：路径上的各 image **本来就是不同几何**（这正是路径的含义），所以这里不比几何指纹；
    # 几何指纹只在「同一结构算了两遍」时才有意义（见 harvest_path.py 的 k1/k2 对照）。
    mismatch = []
    base = next((r for r in rows if r['setup']), None)
    if base is not None:
        for r in rows:
            if r is base or not r['setup']:
                continue
            try:
                q.compare_setups(base['setup'], r['setup'],
                                 'image0(%s)' % os.path.basename(base['file']),
                                 os.path.basename(r['file']), raise_error=True)
            except q.SetupMismatchError as e:
                mismatch.append(str(e))

    # ---- 必修 2：完整性（全在场 + 全收敛 + 全有能量） ----
    absent = list(range(len(files) + 1, expected + 1))      # 1-based，完全没给文件的 image
    missing = [i + 1 for i, r in enumerate(rows) if r['E'] is None]
    unconv = [i + 1 for i, r in enumerate(rows) if r['E'] is not None and not r['conv']]
    complete = (not missing) and (not unconv) and (not absent) and len(rows) >= 2
    rep = dict(rows=rows, complete=complete, missing=missing, unconv=unconv,
               absent=absent, expected=expected, setups_mismatch=mismatch, edef=edef,
               energies_Ry=[r['E'] for r in rows],
               ea_fwd_eV=None, ea_bwd_eV=None, barrier_ok=None, identity=None,
               max_image=None)

    if mismatch:
        rep['error'] = '同一路径上的计算设置不一致，拒绝计算势垒：\n  - ' + \
                       '\n  - '.join(mismatch)
        return rep

    if not complete:
        rep['error'] = ('路径不完整：期望 %d 个 image，实际可用 %d 个；'
                        '缺 image %s；未收敛 image %s'
                        % (expected, len(rows),
                           (absent + missing) or '无', unconv or '无'))
        if not allow_incomplete:
            return rep

    valid = [(r['t'], r['E']) for r in rows if r['E'] is not None]
    if len(valid) < 2:
        rep['error'] = '可用 image 少于 2 个，无法定义路径'
        return rep

    tmax, emax = max(valid, key=lambda x: x[1])
    imax = [i for i, r in enumerate(rows) if r['E'] is not None and r['E'] == emax][0]
    e0, e1 = rows[0]['E'], rows[-1]['E']
    rep['max_image'] = imax
    rep['ea_fwd_eV'] = (emax - e0) * RY_EV if e0 is not None else None
    rep['ea_bwd_eV'] = (emax - e1) * RY_EV if e1 is not None else None
    if e0 is not None and e1 is not None:
        rep['dE_endpoints_eV'] = (e1 - e0) * RY_EV
        res = q.check_energy_identity(rep['ea_fwd_eV'], rep['ea_bwd_eV'],
                                      e1 * RY_EV, e0 * RY_EV, tol=1e-6,
                                      label='路径恒等式', raise_error=False)
        rep['identity'] = res
        rep['barrier_ok'] = res['ok']
        if not res['ok']:
            rep['error'] = '势垒恒等式校验失败：%s' % res.get('message', res)
    return rep


def write_outputs(rep, outdir, edef='E', allow_incomplete=False):
    os.makedirs(outdir, exist_ok=True)
    rows, complete = rep['rows'], rep['complete']
    es = [r['E'] for r in rows if r['E'] is not None]
    ref = min(es) if es else None

    csvp = os.path.join(outdir, 'path_energies.csv')
    with open(csvp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 't', 'E_FplusTS_Ry', 'TS_Ry', 'F_Ry', 'dE_eV', 'M_muB',
                    'scf_status', 'block_index', 'job_done', 'file'])
        for i, r in enumerate(rows):
            de = (r['E'] - ref) * RY_EV if (r['E'] is not None and ref is not None) else None
            w.writerow([i, '%.4f' % r['t'],
                        '%.8f' % r['E'] if r['E'] is not None else 'NA',
                        '%.8f' % r['TS'] if r['TS'] is not None else 'NA',
                        '%.8f' % r['F'] if r['F'] is not None else 'NA',
                        '%.4f' % de if de is not None else 'NA',
                        '%.2f' % r['M'] if r['M'] is not None else 'NA',
                        r['status'], r['block'], r['jobdone'], os.path.basename(r['file'])])

    ndat = os.path.join(outdir, 'neb.dat')
    if complete or allow_incomplete:
        with open(ndat, 'w', encoding='utf-8') as f:
            f.write('# t  E(%s)-ref [eV]   F-ref [eV]   TS_Ry\n' % edef)
            for r in rows:
                if r['E'] is None:
                    continue
                f.write('%.4f  %.5f  %.5f  %s\n'
                        % (r['t'], (r['E'] - ref) * RY_EV,
                           (r['F'] - ref) * RY_EV if r['F'] is not None else float('nan'),
                           'NA' if r['TS'] is None else '%.8f' % r['TS']))

    summary = []
    summary.append('images: %d (valid %d, 期望 %d)   E 口径: %s   %s'
                   % (len(rows), len(es), rep.get('expected', len(rows)), edef,
                      '完整' if complete else '不完整（诊断剖面，非最终势垒）'))
    for r in rows:
        if r['E'] is None:
            summary.append('  t=%.3f  MISSING  status=%s  %s'
                           % (r['t'], r['status'], os.path.basename(r['file'])))
            continue
        summary.append('  t=%.3f  %s=%.8f Ry  TS=%s Ry  dE=%.3f eV  M=%s  status=%s'
                       % (r['t'], edef, r['E'],
                          'NA' if r['TS'] is None else '%.8f' % r['TS'],
                          (r['E'] - ref) * RY_EV,
                          'NA' if r['M'] is None else '%.2f' % r['M'], r['status']))
    for j in rep.get('absent', []):
        summary.append('  image %d  MISSING（未提供文件）' % j)
    if ref is not None:
        summary.append('reference E_ref = %.8f Ry（%s 口径，取全部 image 最低）' % (ref, edef))
    if 'error' in rep:
        summary.append('ERROR: ' + rep['error'])
    if rep.get('dE_endpoints_eV') is not None:
        summary.append('E(final)-E(initial) = %+.4f eV' % rep['dE_endpoints_eV'])
    if rep.get('ea_fwd_eV') is not None:
        summary.append('barrier forward  (initial -> img%d) = %.4f eV'
                       % (rep['max_image'], rep['ea_fwd_eV']))
        summary.append('barrier backward (final   -> img%d) = %.4f eV'
                       % (rep['max_image'], rep['ea_bwd_eV']))
        if rep.get('identity'):
            summary.append('identity  Ea(->)-Ea(<-) = E_final-E_initial : %s (diff=%.2e eV, tol=1e-6)'
                           % ('OK' if rep['identity']['ok'] else 'FAIL', rep['identity']['diff']))
    summary.append('NOTE: 静态线性路径（未做 NEB 弛豫）——是势垒上界，不是 NEB 结果。')
    txt = '\n'.join(summary)
    with open(os.path.join(outdir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    return txt, csvp, ndat


def main(argv=None):
    ap = argparse.ArgumentParser(description='路径能量剖面与势垒（按计算块解析，缺 image 即报不完整）')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--nimg', type=int, default=None,
                    help='期望的 image 数（仅用于核对；实际以文件数为准）')
    ap.add_argument('--edef', choices=['E', 'F'], default='E',
                    help='能量口径：E=F+TS（默认）或 F')
    ap.add_argument('--allow-incomplete', action='store_true',
                    help='缺 image/未收敛时仍写诊断剖面（明确标注，不作为最终势垒）')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args(argv)

    rep = analyze(a.files, edef=a.edef, allow_incomplete=a.allow_incomplete, nimg=a.nimg)
    txt, csvp, ndat = write_outputs(rep, a.outdir, edef=a.edef,
                                    allow_incomplete=a.allow_incomplete)
    print(txt)
    if 'error' in rep:
        print('\n!! ' + rep['error'])
        if not a.allow_incomplete:
            print('!! 未写 neb.dat；加 --allow-incomplete 可输出诊断剖面')
            return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
