"""QA strips: for selected Gold indices, show 5 consecutive sagittal slices around the estimated notch (notch-12 .. notch+12 mm) as 110 mm crops."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compartment_locator_v1 as CL, ligament_locator_v1 as LL

if __name__ == '__main__':
    idx = [int(x) for x in sys.argv[1].split(',')]; tag = sys.argv[2]
    rows = {json.loads(l)['gold_index']: json.loads(l) for l in open(os.path.join(LL.OUT, 'coverage.jsonl'))}
    gold = pd.read_csv(os.path.join(CL.W, 'data/gold58_mri/gold58_labels.csv'), dtype=str).StudyInstanceUID.tolist()
    fig, axes = plt.subplots(len(idx), 5, figsize=(17, 3.5 * len(idx)))
    for row, gi in enumerate(idx):
        r = rows[gi]; uid = gold[gi - 1]; series = CL.load_study(uid); s = series[r['notch_series']]
        xs = np.array([LL.slice_center(d, 0) for d in s]); order = np.argsort(np.abs(xs - r['notch_x']))[:9]; picks = sorted(order, key=lambda k: xs[k])
        # 5 evenly spaced among the 9 nearest
        sel = [picks[i] for i in np.linspace(0, len(picks) - 1, 5).round().astype(int)]
        for col, k in enumerate(sel):
            ax = axes[row, col] if len(idx) > 1 else axes[col]; ax.axis('off')
            t = LL.resample(s[k], (1, 2), r['sagn_y0'], r['sagn_y1'], r['sagn_z0'], r['sagn_z1'])
            ax.imshow(t, cmap='gray', vmin=np.percentile(t, 1), vmax=np.percentile(t, 99.5)); ax.axhline(r['sagn_z1'] - r['joint_z'], color='y', lw=0.6)
            ax.set_title(f"g{gi} {r['side']} dx={xs[k] - r['notch_x']:+.0f}mm", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(LL.OUT, f'strip_{tag}.png'), dpi=80); plt.close(fig); print('saved', tag)
