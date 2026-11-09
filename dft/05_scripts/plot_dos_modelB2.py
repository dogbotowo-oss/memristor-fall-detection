#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot spin-resolved DOS / PDOS for the actual-composition model (bulk vs O27 vacancy).

Reads the aggregated files produced by sum_pdos_new.py / dos.x and draws PNGs with
PIL (matplotlib is not available). Spin-up is drawn upwards, spin-down downwards.

Usage: python3 plot_dos_modelB2.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DOSDIR = os.path.join(ROOT, '03_DOS', 'modelB')


def load_dos(fn):
    rows = []
    for ln in open(fn):
        if ln.startswith('#'):
            continue
        p = ln.split()
        if len(p) < 3:
            continue
        rows.append((float(p[0]), float(p[1]), float(p[2])))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def load_chan(fn):
    with open(fn) as f:
        hdr = f.readline().lstrip('#').split()
        arr = np.loadtxt(f)
    return hdr, arr


def smooth(y, n=5):
    if n <= 1:
        return y
    k = np.ones(n) / n
    return np.convolve(y, k, mode='same')


def font(size=16):
    for name in ('arial.ttf', 'segoeui.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


class Plot:
    def __init__(self, w, h, xlim, ylim, title, xlabel, ylabel, note=None):
        self.w, self.h = w, h
        self.xlim, self.ylim = xlim, ylim
        self.img = Image.new('RGB', (w, h), 'white')
        self.d = ImageDraw.Draw(self.img)
        self.l, self.r, self.t, self.b = 95, 40, 62, 62
        self.f = font(16)
        self.fs = font(13)
        self.title, self.xlabel, self.ylabel, self.note = title, xlabel, ylabel, note

    def xy(self, x, y):
        px = self.l + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.w - self.l - self.r)
        py = self.b + (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.h - self.t - self.b)
        return px, self.h - py

    def axes(self):
        d = self.d
        d.rectangle([self.l, self.t, self.w - self.r, self.h - self.b], outline='black', width=2)
        x0, y0 = self.xy(0.0, 0.0)
        d.line([self.l, y0, self.w - self.r, y0], fill=(120, 120, 120), width=1)
        if self.xlim[0] < 0 < self.xlim[1]:
            d.line([x0, self.t, x0, self.h - self.b], fill=(200, 60, 60), width=2)
            d.text((x0 + 4, self.b - 2), 'E_F', fill=(200, 60, 60), font=self.fs)
        d.text((self.l, 10), self.title, fill='black', font=self.f)
        d.text((self.w // 2 - 70, self.h - 26), self.xlabel, fill='black', font=self.fs)
        d.text((8, self.t - 34), self.ylabel, fill='black', font=self.fs)
        if self.note:
            d.text((self.l + 10, self.h - 52), self.note, fill=(60, 60, 60), font=self.fs)

    def curve(self, x, y, color, width=2):
        pts = [self.xy(float(a), float(b)) for a, b in zip(x, y) if self.xlim[0] <= a <= self.xlim[1]]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width)

    def legend(self, items, pos):
        x, y = pos
        for lab, col in items:
            self.d.line([x, y + 7, x + 26, y + 7], fill=col, width=3)
            self.d.text((x + 32, y), lab, fill='black', font=self.fs)
            y += 22

    def save(self, fn):
        self.img.save(fn)
        print('saved', fn)


def main():
    EFb, EFv = 10.487, 10.730
    Eb, ub, db = load_dos(os.path.join(DOSDIR, 'dos_bulk.dat'))
    Ev, uv, dv = load_dos(os.path.join(DOSDIR, 'dos_vacO27.dat'))

    winb = (Eb - EFb > -9) & (Eb - EFb < 9)
    winv = (Ev - EFv > -9) & (Ev - EFv < 9)
    m = max(np.max(np.abs(ub[winb])), np.max(np.abs(db[winb])),
            np.max(np.abs(uv[winv])), np.max(np.abs(dv[winv])))
    p = Plot(1100, 700, (-9, 9), (-1.12 * m, 1.12 * m),
             'Spin-resolved total DOS: bulk (blue) vs V_O(O27) (red); E_F = 0',
             'E - E_F (eV)', 'DOS (states/eV)',
             'DOS at E_F: bulk 0.73, V_O(O27) 3.86 states/eV')
    p.axes()
    p.curve(Eb - EFb, smooth(ub, 7), (30, 90, 200), 2)
    p.curve(Eb - EFb, -smooth(db, 7), (30, 90, 200), 2)
    p.curve(Ev - EFv, smooth(uv, 7), (200, 60, 60), 2)
    p.curve(Ev - EFv, -smooth(dv, 7), (200, 60, 60), 2)
    p.legend([('bulk (up / down)', (30, 90, 200)), ('V_O O27 (up / down)', (200, 60, 60))],
             (p.l + 40, p.t + 24))
    p.save(os.path.join(DOSDIR, 'dos_bulk_vs_vacO27.png'))

    hb, ab = load_chan(os.path.join(DOSDIR, 'pdos_bulk_by_channel.dat'))
    hv, av = load_chan(os.path.join(DOSDIR, 'pdos_vacO27_by_channel.dat'))
    col = {n: i for i, n in enumerate(hb)}
    panels = [('FeA_d', 'FeA 3d'), ('FeB_d', 'FeB 3d'), ('Ni_d', 'Ni 3d'),
              ('Cu_d', 'Cu 3d'), ('Zn_d', 'Zn 3d'), ('O_p', 'O 2p')]
    for key, title in panels:
        cu, cd = key + '_up', key + '_down'
        if cu not in col:
            continue
        wb = (ab[:, 0] - EFb > -9) & (ab[:, 0] - EFb < 9)
        wv = (av[:, 0] - EFv > -9) & (av[:, 0] - EFv < 9)
        mmax = max(np.max(np.abs(ab[wb, col[cu]])), np.max(np.abs(av[wv, col[cu]])),
                   np.max(np.abs(ab[wb, col[cd]])), np.max(np.abs(av[wv, col[cd]])))
        p = Plot(900, 560, (-9, 9), (-1.15 * mmax, 1.15 * mmax),
                 'PDOS %s: bulk (blue) vs V_O(O27) (red); E_F = 0' % title,
                 'E - E_F (eV)', 'PDOS (states/eV)')
        p.axes()
        p.curve(ab[:, 0] - EFb, smooth(ab[:, col[cu]], 7), (30, 90, 200), 2)
        p.curve(ab[:, 0] - EFb, -smooth(ab[:, col[cd]], 7), (30, 90, 200), 2)
        p.curve(av[:, 0] - EFv, smooth(av[:, col[cu]], 7), (200, 60, 60), 2)
        p.curve(av[:, 0] - EFv, -smooth(av[:, col[cd]], 7), (200, 60, 60), 2)
        p.legend([('bulk', (30, 90, 200)), ('V_O O27', (200, 60, 60))], (p.l + 20, p.t + 24))
        p.save(os.path.join(DOSDIR, 'pdos_%s_bulk_vs_vac.png' % key))

    print('done')


main()
