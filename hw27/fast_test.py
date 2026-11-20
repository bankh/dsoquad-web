#!/usr/bin/env python3
"""36 MS/s and 72 MS/s (interleaved) test for the HW 2.7 report.

Wiring: Wave Out -> channel A. The generator runs a 1 MHz square wave.

Checks, at 36 MS/s and at 72 MS/s: the frame length and flags, the number of samples per
cycle (36 and 72 expected), and the zigzag between neighbouring samples on the flat parts,
which shows whether the two interleaved converters are lined up.

    python3 hw27/fast_test.py
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
from protocol import ACQ_AUTO, ACQ_STOP  # noqa: E402

SKIP = 16


async def run(a):
    dev = await Device().open(a.port)
    got = {}
    try:
        fw = (await dev.hello()).get('fw')
        await dev.set_channel(0, a.range, 0, a.offset)
        await dev.set_channel(1, a.range, 0, a.offset)
        await dev.set_gen(1, a.gen, 50)
        await dev.set_trigger(0, 1, a.offset + 40)
        for rate in (36_000_000, 72_000_000):
            await dev.set_rate(rate)
            await asyncio.sleep(1.0)
            st = await dev.state()
            while not dev.frames.empty():
                dev.frames.get_nowait()
            await dev.set_acq(ACQ_AUTO, 200)
            try:
                frames = [await dev.next_frame(timeout=5) for _ in range(4)]
            finally:
                await dev.set_acq(ACQ_STOP)
            f = frames[-1]
            got[rate] = {'rate_actual': st.rate_actual, 'frame_rate': f.rate, 'interleaved': f.interleaved,
                         'triggered': f.triggered, 'raw_a_len': len(f.a),
                         'x': np.array(f.channel_a(), dtype=float)[SKIP:]}
        await dev.set_rate(100_000)
        await dev.set_gen(1, 1000, 50)
    finally:
        await dev.close()
    return fw, got


def analyse(x, rate, gen):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    mid = (lo + hi) / 2
    ab = x >= mid
    rising = np.flatnonzero(~ab[:-1] & ab[1:]) + 1
    per = np.diff(rising)
    flat_hi = x[x > hi - (hi - lo) * 0.15]
    # zigzag: mean difference between even- and odd-indexed samples on the high plateau
    idx = np.flatnonzero(x > hi - (hi - lo) * 0.15)
    even, odd = x[idx[idx % 2 == 0]], x[idx[idx % 2 == 1]]
    # rise: samples between 10% and 90% on the first clean rising edge
    r0 = rising[2] if len(rising) > 2 else None
    edge = x[r0 - 6:r0 + 10].round(1).tolist() if r0 else None
    return {'samples': int(x.size), 'low_code': round(float(lo), 1), 'high_code': round(float(hi), 1),
            'cycles': int(per.size), 'samples_per_cycle_expected': rate / gen,
            'samples_per_cycle_mean': round(float(per.mean()), 3) if per.size else None,
            'samples_per_cycle_counts': {int(v): int(c) for v, c in zip(*np.unique(per, return_counts=True))},
            'measured_hz': round(float(rate / per.mean())) if per.size else None,
            'plateau_noise_std_codes': round(float(flat_hi.std()), 2),
            'even_minus_odd_on_plateau_codes': round(float(even.mean() - odd.mean()), 2),
            'edge_samples': edge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--range', type=int, default=3)
    ap.add_argument('--offset', type=int, default=64)
    ap.add_argument('--gen', type=float, default=1_000_000)
    ap.add_argument('--name', default='fast')
    a = ap.parse_args()
    fw, got = asyncio.run(run(a))
    res = HERE / 'results'
    res.mkdir(exist_ok=True)
    out = {'fw': fw, 'gen_hz': a.gen}
    for rate, g in got.items():
        x = g.pop('x')
        g.update(analyse(x, g['frame_rate'] or rate, a.gen))
        out[f'{rate // 1_000_000}MS/s'] = g
        got[rate]['x'] = x
    np.savez_compressed(res / f'{a.name}.npz', **{f'a_{r // 1_000_000}M': g['x'] for r, g in got.items()})
    printable = {k: ({kk: vv for kk, vv in v.items() if kk != 'x'} if isinstance(v, dict) else v) for k, v in out.items()}
    (res / f'{a.name}.json').write_text(json.dumps(printable, indent=1))
    print(json.dumps(printable, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for row, (rate, g) in enumerate(got.items()):
        x = g['x']
        fr = g['frame_rate'] or rate
        n = int(fr / a.gen * 4)
        ax[row, 0].plot(np.arange(n) / fr * 1e6, x[200:200 + n], '.-', ms=3, lw=0.8)
        ax[row, 0].set(title=f"{fr / 1e6:g} MS/s{' (interleaved)' if g['interleaved'] else ''}: "
                             f"{a.gen / 1e6:g} MHz square wave, {g['samples']} samples",
                       xlabel='time (µs)', ylabel='channel A (ADC code)')
        ab = x >= (np.percentile(x, 2) + np.percentile(x, 98)) / 2
        r = np.flatnonzero(~ab[:-1] & ab[1:]) + 1
        for e in r[2:40]:
            ax[row, 1].plot(np.arange(-6, 12) / fr * 1e9, x[e - 6:e + 12], '.-', ms=3, lw=0.5, alpha=0.5)
        ax[row, 1].set(title='Rising edges overlaid', xlabel='time from threshold crossing (ns)', ylabel='ADC code')
    for x_ in ax.flat:
        x_.grid(alpha=0.3)
    fig.suptitle(f'Fast sampling, HW 2.7, fw {fw}')
    fig.tight_layout()
    fig.savefig(res / f'{a.name}.png', dpi=110)


if __name__ == '__main__':
    main()
