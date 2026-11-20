#!/usr/bin/env python3
"""Digital channels C and D test for the HW 2.7 report.

Wiring: Wave Out -> channel A (reference) AND channel C AND channel D.

1. Square wave: C and D should toggle with A, edge for edge.
2. Slow triangle from the DAC: channel A measures the voltage at which C and D switch.

    python3 hw27/cd_test.py
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'dsoq'))
from device import Device  # noqa: E402
from protocol import ACQ_AUTO, ACQ_STOP, CODES_PER_DIV  # noqa: E402

SKIP = 8


async def grab(dev, n=1):
    while not dev.frames.empty():
        dev.frames.get_nowait()
    await dev.set_acq(ACQ_AUTO, 200)
    try:
        frames = [await dev.next_frame(timeout=5) for _ in range(n + 1)]
    finally:
        await dev.set_acq(ACQ_STOP)
    return frames[1:]


def unpack(frames):
    a = np.concatenate([np.frombuffer(bytes(f.a), np.uint8)[SKIP:] for f in frames]).astype(int)
    cd = np.concatenate([np.frombuffer(bytes(f.cd), np.uint8)[SKIP:] for f in frames]).astype(int)
    return a, cd & 1, (cd >> 1) & 1


FRAME = 4096 - SKIP  # samples kept per frame; several frames are stored back to back


def edges(x, thr=None):
    ab = x >= ((x.min() + x.max()) / 2 if thr is None else thr)
    i = np.flatnonzero(ab[:-1] != ab[1:]) + 1
    i = i[i % FRAME != 0]  # a level change exactly where two frames meet is not a real edge
    return i, ab[i]


async def run(a):
    dev = await Device().open(a.port)
    try:
        fw = (await dev.hello()).get('fw')
        tables = await dev.tables()
        await dev.set_channel(0, a.range, 0, a.offset)
        await dev.set_channel(1, a.range, 0, a.offset)
        await dev.set_gen(1, 1000, 50)
        await dev.set_rate(a.rate)
        await dev.set_trigger(0, 1, a.offset + 40)
        await asyncio.sleep(1.5)
        sq = unpack(await grab(dev, 4))
        n = 512
        up = np.linspace(100, 4000, n // 2)
        await dev.set_gen(0, 1000, 50)
        await dev.set_wave(np.concatenate([up, up[::-1]]).astype(int).tolist())
        await dev.set_gen(2, 50, 50)
        await dev.set_rate(100_000)
        await asyncio.sleep(1.5)
        tri = unpack(await grab(dev, 8))
        await dev.set_gen(1, 1000, 50)
    finally:
        await dev.close()
    return fw, tables, sq, tri


def plot(a, fw, sa, sc, sd, ta, tc, td, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    n = min(int(a.rate * 0.004), FRAME)  # 4 ms, or one whole frame when that is shorter
    t = np.arange(n) / a.rate * 1e3
    lo, hi = sa.min(), sa.max()
    ax[0].plot(t, (sa[:n] - lo) / (hi - lo), label='A (scaled 0 to 1)')
    ax[0].step(t, sc[:n] * 0.8 - 1.2, where='post', label='C (digital)')
    ax[0].step(t, sd[:n] * 0.8 - 2.4, where='post', label='D (digital)')
    ax[0].set(title=f'1 kHz square wave on A, C and D ({a.rate / 1e6:g} MS/s, one frame), HW 2.7, fw {fw}', xlabel='time (ms)', yticks=[])
    m = min(FRAME, ta.size)
    t2 = np.arange(m) / 100_000 * 1e3
    ax[1].plot(t2, ta[:m], label='A (ADC code)')
    span = ta.max() - ta.min()
    ax[1].step(t2, tc[:m] * span * 0.3 + ta.min() - span * 0.45, where='post', label='C (digital)')
    ax[1].step(t2, td[:m] * span * 0.3 + ta.min() - span * 0.9, where='post', label='D (digital)')
    ax[1].set(title='50 Hz triangle: where C and D switch along the ramp', xlabel='time (ms)', ylabel='ADC code (A)')
    for x_ in ax:
        x_.grid(alpha=0.3)
        x_.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--range', type=int, default=3)
    ap.add_argument('--offset', type=int, default=64)
    ap.add_argument('--rate', type=float, default=1_000_000)
    ap.add_argument('--volts-per-div', type=float, default=0.5, help='of --range, for the threshold estimate')
    ap.add_argument('--name', default='cd')
    ap.add_argument('--replot', action='store_true', help='redraw the plot from saved results, no scope needed')
    a = ap.parse_args()
    res = HERE / 'results'
    res.mkdir(exist_ok=True)
    if a.replot:
        d = np.load(res / f'{a.name}.npz')
        j = json.loads((res / f'{a.name}.json').read_text())
        a.rate = j['square_rate']
        plot(a, j['fw'], d['sq_a'], d['sq_c'], d['sq_d'], d['tri_a'], d['tri_c'], d['tri_d'], res / f'{a.name}.png')
        return
    fw, tables, (sa, sc, sd), (ta, tc, td) = asyncio.run(run(a))
    out = {'fw': fw, 'square_rate': a.rate, 'square': {}, 'threshold': {}}
    ea, pa = edges(sa)
    for name, x in (('C', sc), ('D', sd)):
        e, p = edges(x, 0.5)
        o = {'low_samples': int(np.sum(x == 0)), 'high_samples': int(np.sum(x == 1)),
             'edges': int(len(e)), 'edges_a': int(len(ea))}
        if len(e) == len(ea) and len(e):
            d = e - ea
            o['edge_offset_vs_a_samples'] = {int(v): int(c) for v, c in zip(*np.unique(d, return_counts=True))}
            o['same_polarity_as_a'] = bool(np.all(p == pa))
        out['square'][name] = o
    volts = lambda code: (code - a.offset) / CODES_PER_DIV * a.volts_per_div  # noqa: E731
    for name, x in (('C', tc), ('D', td)):
        e, p = edges(x, 0.5)
        if len(e):
            up_codes, dn_codes = ta[e][p], ta[e][~p]
            out['threshold'][name] = {
                'transitions': int(len(e)),
                'switches_high_at_code': [float(np.mean(up_codes)), float(np.std(up_codes))] if len(up_codes) else None,
                'switches_low_at_code': [float(np.mean(dn_codes)), float(np.std(dn_codes))] if len(dn_codes) else None,
                'switches_high_at_volts_uncalibrated': round(volts(float(np.mean(up_codes))), 2) if len(up_codes) else None,
                'switches_low_at_volts_uncalibrated': round(volts(float(np.mean(dn_codes))), 2) if len(dn_codes) else None}
        else:
            out['threshold'][name] = {'transitions': 0, 'stuck_at': int(x[0])}
    out['a_range_codes_triangle'] = [int(ta.min()), int(ta.max())]
    np.savez_compressed(res / f'{a.name}.npz', sq_a=sa, sq_c=sc, sq_d=sd, tri_a=ta, tri_c=tc, tri_d=td)
    (res / f'{a.name}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    plot(a, fw, sa, sc, sd, ta, tc, td, res / f'{a.name}.png')

if __name__ == '__main__':
    main()
