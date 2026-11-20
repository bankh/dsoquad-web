#!/usr/bin/env python3
"""Channel A vs channel B test for the HW 2.7 report.

Wiring: Wave Out -> channel A AND channel B (same signal on both).

1. Square wave: both channels should show the same edges at the same samples.
2. Slow triangle from the DAC: a smooth ramp. The firmware swaps the two low bits of channel B
   (a correction written for FPGA 2.61). On a ramp a wrong correction shows as a zigzag, so the
   script compares B as delivered with B after undoing the swap and reports which is smoother.

    python3 hw27/ab_test.py
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

SKIP = 8  # first samples of a frame are stale FIFO contents (docs/protocol.md)


def swap_low_bits(x):
    x = x.copy()
    low = x & 3
    m = (low == 1) | (low == 2)
    x[m] ^= 3
    return x


def roughness(x):
    """Mean absolute second difference: small for a smooth ramp, larger for a zigzag."""
    return float(np.mean(np.abs(np.diff(x.astype(int), 2))))


async def grab(dev, n=1):
    while not dev.frames.empty():
        dev.frames.get_nowait()
    await dev.set_acq(ACQ_AUTO, 200)
    try:
        frames = [await dev.next_frame(timeout=5) for _ in range(n + 1)]
    finally:
        await dev.set_acq(ACQ_STOP)
    return frames[1:]  # drop the first frame: it may predate the settings


async def run(a):
    dev = await Device().open(a.port)
    out = {}
    try:
        out['fw'] = (await dev.hello()).get('fw')
        for ch in (0, 1):
            await dev.set_channel(ch, a.range, 0, a.offset)

        # 1. square wave
        await dev.set_gen(1, 1000, 50)
        await dev.set_rate(100_000)
        await dev.set_trigger(0, 1, a.offset + 40)
        await asyncio.sleep(1.5)
        f = (await grab(dev))[0]
        sq = (np.frombuffer(bytes(f.a), np.uint8)[SKIP:].astype(int),
              np.frombuffer(bytes(f.b), np.uint8)[SKIP:].astype(int), f.rate)

        # 2. slow triangle from the DAC
        n = 512
        up = np.linspace(200, 3900, n // 2)
        table = np.concatenate([up, up[::-1]]).astype(int).tolist()
        await dev.set_gen(0, 1000, 50)
        await dev.set_wave(table)
        await dev.set_gen(2, a.tri_hz, 50)
        await dev.set_rate(a.tri_rate)
        await asyncio.sleep(1.5)
        fs = await grab(dev, a.frames)
        tri = (np.concatenate([np.frombuffer(bytes(f.a), np.uint8)[SKIP:] for f in fs]).astype(int),
               np.concatenate([np.frombuffer(bytes(f.b), np.uint8)[SKIP:] for f in fs]).astype(int),
               fs[0].rate, len(fs[0].a) - SKIP)
        await dev.set_gen(1, 1000, 50)
    finally:
        await dev.close()
    return out, sq, tri


def edges(x):
    mid = (x.min() + x.max()) / 2
    ab = x >= mid
    return np.flatnonzero(ab[:-1] != ab[1:]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--range', type=int, default=3)
    ap.add_argument('--offset', type=int, default=64)
    ap.add_argument('--tri-hz', type=float, default=50)
    ap.add_argument('--tri-rate', type=float, default=100_000)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--name', default='ab')
    a = ap.parse_args()
    out, (sa, sb, srate), (ta, tb, trate, flen) = asyncio.run(run(a))
    res = HERE / 'results'
    res.mkdir(exist_ok=True)

    ea, eb = edges(sa), edges(sb)
    same = len(ea) == len(eb)
    out['square'] = {'rate': srate, 'a_codes': [int(sa.min()), int(sa.max())], 'b_codes': [int(sb.min()), int(sb.max())],
                     'edges_a': int(len(ea)), 'edges_b': int(len(eb)),
                     'edge_offset_samples_b_minus_a': (sorted({int(v) for v in (eb - ea)}) if same else None),
                     'mean_abs_diff_codes': float(np.mean(np.abs(sa - sb)))}

    tb_un = swap_low_bits(tb)
    low = lambda x: [int(np.sum((x & 3) == k)) for k in range(4)]  # noqa: E731
    out['triangle'] = {
        'rate': trate, 'hz': a.tri_hz, 'samples': int(ta.size),
        'a_codes': [int(ta.min()), int(ta.max())], 'b_codes': [int(tb.min()), int(tb.max())],
        'roughness_a': roughness(ta),
        'roughness_b_as_delivered': roughness(tb),
        'roughness_b_swap_undone': roughness(tb_un),
        'low2bits_hist_a': low(ta), 'low2bits_hist_b': low(tb),
        'a_minus_b_mean': float(np.mean(ta - tb)), 'a_minus_b_std': float(np.std(ta - tb)),
        'a_minus_b_std_swap_undone': float(np.std(ta - tb_un)),
        'distinct_codes_a': int(np.unique(ta).size), 'distinct_codes_b': int(np.unique(tb).size),
    }
    t = out['triangle']
    t['verdict'] = ('correction looks right (B is smoother as delivered)'
                    if t['roughness_b_as_delivered'] <= t['roughness_b_swap_undone']
                    else 'correction looks WRONG on this unit (B is smoother with the swap undone)')

    np.savez_compressed(res / f'{a.name}.npz', sq_a=sa, sq_b=sb, tri_a=ta, tri_b=tb)
    (res / f'{a.name}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    n = 400
    ax[0, 0].plot(np.arange(n) / srate * 1e3, sa[:n], label='A')
    ax[0, 0].plot(np.arange(n) / srate * 1e3, sb[:n], '--', label='B')
    ax[0, 0].set(title='1 kHz square wave on A and B (100 kS/s)', xlabel='time (ms)', ylabel='ADC code')
    m = min(flen, ta.size)
    ax[0, 1].plot(np.arange(m) / trate * 1e3, ta[:m], label='A')
    ax[0, 1].plot(np.arange(m) / trate * 1e3, tb[:m], label='B')
    ax[0, 1].set(title=f'{a.tri_hz:g} Hz triangle from the DAC on A and B', xlabel='time (ms)', ylabel='ADC code')
    # zoom on a rising part of the ramp
    i0 = int(np.argmin(ta[:m // 2])) + 50
    z = slice(i0, i0 + 220)
    x = np.arange(z.start, z.stop) / trate * 1e3
    ax[1, 0].step(x, ta[z], where='mid', label='A')
    ax[1, 0].step(x, tb[z], where='mid', label='B as delivered')
    ax[1, 0].step(x, tb_un[z] - 6, where='mid', label='B with swap undone (shifted down 6)')
    ax[1, 0].set(title='Zoom on the ramp: which version of B is smooth?', xlabel='time (ms)', ylabel='ADC code')
    w = 0.27
    k = np.arange(4)
    ax[1, 1].bar(k - w, t['low2bits_hist_a'], w, label='A')
    ax[1, 1].bar(k, t['low2bits_hist_b'], w, label='B as delivered')
    ax[1, 1].bar(k + w, low(tb_un), w, label='B swap undone')
    ax[1, 1].set(title='How often each value of the two low bits occurs', xlabel='code mod 4', ylabel='samples', xticks=k)
    for x_ in ax.flat:
        x_.grid(alpha=0.3)
        x_.legend(fontsize=8)
    fig.suptitle(f"A vs B, HW 2.7, fw {out['fw']}: {t['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(res / f'{a.name}.png', dpi=110)


if __name__ == '__main__':
    main()
