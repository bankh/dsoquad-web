#!/usr/bin/env python3
"""Roll-mode (continuous recording) test for the HW 2.7 report.

Wiring: Wave Out -> channel A. The generator runs a square wave; the script records in roll
mode for a while, checks that the stream has no gaps, and measures the period of every cycle.

    python3 hw27/roll_test.py --rate 50000 --seconds 60 --name roll_50k
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'dsoq'))
from device import Device  # noqa: E402
from protocol import ACQ_ROLL, ACQ_STOP  # noqa: E402


async def record(a):
    dev = await Device().open(a.port)
    try:
        hello = await dev.hello()
        await dev.set_gen(1, a.gen, 50)
        await dev.set_rate(a.rate)
        await dev.set_channel(0, a.range_a, 0, a.offset_a)
        await dev.set_channel(1, a.range_a, 0, a.offset_a)
        await asyncio.sleep(1.0)  # let the input settle after the range/offset change
        st = await dev.state()
        await dev.set_acq(ACQ_ROLL)
        chunks, jumps, gap_flags, expect = [], 0, 0, None
        host_t = []
        t0 = time.time()
        try:
            while time.time() - t0 < a.seconds:
                c = await asyncio.wait_for(dev.rolls.get(), 5.0)
                if expect is not None and c.frame_no != expect:
                    jumps += 1
                gap_flags += bool(c.flags & 0x10)
                expect = c.frame_no + len(c.a)
                chunks.append(c)
                host_t.append(time.time() - t0)
        finally:
            await dev.set_acq(ACQ_STOP)
        dt = time.time() - t0
    finally:
        await dev.close()
    # The first chunk after arming carries the gap flag by design (docs/protocol.md).
    sa = np.frombuffer(b''.join(bytes(c.a) for c in chunks), dtype=np.uint8)
    sb = np.frombuffer(b''.join(bytes(c.b) for c in chunks), dtype=np.uint8)
    scd = np.frombuffer(b''.join(bytes(c.cd) for c in chunks), dtype=np.uint8)
    meta = {'fw': hello.get('fw'), 'rate_requested': a.rate, 'rate_actual': st.rate_actual,
            'gen_hz': a.gen, 'seconds': round(dt, 2), 'chunks': len(chunks), 'samples': int(sa.size),
            'samples_per_s': round(sa.size / dt), 'index_jumps': jumps, 'gap_flags': gap_flags,
            'chunk_sizes': sorted({len(c.a) for c in chunks}),
            'first_index': chunks[0].frame_no, 'last_index_end': expect}
    return meta, sa, sb, scd, np.array(host_t)


def analyse(meta, sa):
    rate = meta['rate_actual']
    lo, hi = int(sa.min()), int(sa.max())
    mid = (lo + hi) / 2
    above = sa >= mid
    rising = np.flatnonzero(~above[:-1] & above[1:]) + 1
    periods = np.diff(rising)
    nominal = rate / meta['gen_hz']
    res = {'code_min': lo, 'code_max': hi, 'cycles': int(periods.size),
           'period_nominal_samples': nominal}
    if periods.size:
        vals, counts = np.unique(periods, return_counts=True)
        res.update({'period_min': int(periods.min()), 'period_max': int(periods.max()),
                    'period_mean': float(periods.mean()),
                    'period_counts': {int(v): int(c) for v, c in zip(vals, counts)},
                    'cycles_off_by_more_than_1_sample': int(np.sum(np.abs(periods - nominal) > 1)),
                    'measured_hz': float(rate / periods.mean())})
    return res, rising, periods


def plot(meta, res, sa, rising, periods, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rate = meta['rate_actual']
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    n = min(sa.size, int(rate * 0.01))
    ax[0].plot(np.arange(n) / rate * 1e3, sa[:n], lw=0.8)
    ax[0].set(xlabel='time (ms)', ylabel='channel A (ADC code)',
              title=f"Roll mode, HW 2.7, fw {meta['fw']}: {meta['samples']} samples in {meta['seconds']} s "
                    f"at {rate / 1e3:g} kS/s, {meta['index_jumps']} index jumps")
    if periods.size:
        ax[1].plot(rising[1:] / rate, periods / rate * 1e6, '.', ms=2)
        ax[1].set(xlabel='time (s)', ylabel='cycle period (µs)',
                  title=f"Period of each of {periods.size} cycles (nominal {1e6 / meta['gen_hz']:g} µs, "
                        f"one sample = {1e6 / rate:g} µs)")
        vals, counts = np.unique(periods, return_counts=True)
        ax[2].bar(vals / rate * 1e6, counts, width=0.8e6 / rate)
        ax[2].set(xlabel='cycle period (µs)', ylabel='number of cycles (log scale)', yscale='log',
                  title='Distribution of cycle periods')
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--rate', type=float, default=50_000)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--gen', type=float, default=1000)
    ap.add_argument('--range-a', type=int, default=4)
    ap.add_argument('--offset-a', type=int, default=104)
    ap.add_argument('--name', default='roll')
    ap.add_argument('--out', default=str(HERE / 'results'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, sa, sb, scd, host_t = asyncio.run(record(a))
    res, rising, periods = analyse(meta, sa)
    np.savez_compressed(out / f'{a.name}.npz', a=sa, b=sb, cd=scd, chunk_host_time_s=host_t)
    plot(meta, res, sa, rising, periods, out / f'{a.name}.png')
    summary = {'test': 'roll', **meta, **res}
    (out / f'{a.name}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
