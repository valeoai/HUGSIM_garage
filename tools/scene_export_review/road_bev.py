"""Top-down review of one scene's exported road map against its Gaussians.

Writes <out>.png (left: ground colour orthophoto, right: semantic orthophoto, both with the
exported EDGE / LINE / LANE / parked cars and a 10 m grid in sim coordinates) and <out>.json
(agreement of the raster drivable area with the Gaussians' road class near the drive).

usage: road_bev.py <scene_dir> <out_prefix> [roads_override.npy]
"""
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

RES = 0.2          # m per pixel
MARGIN = 35.0      # m around the drive
NEAR = 25.0        # m from the drive, for the metric
ROAD, KERB = (0,), (1, 9)
SEM_COL = {0: (128, 128, 128), 1: (230, 120, 200), 9: (90, 170, 60), 8: (20, 90, 20),
           13: (40, 80, 230), 14: (40, 80, 230), 15: (40, 80, 230), 2: (120, 90, 60)}


def load(scene):
    mp, _ = torch.load(scene + "/scene.pth", weights_only=False, map_location="cpu")
    parts = [mp, mp[10]] if len(mp) > 10 and isinstance(mp[10], tuple) else [mp]  # scene + ground model
    xyz = np.concatenate([p[1].detach().numpy() for p in parts]).astype(np.float64)
    col = np.clip(0.28209479 * np.concatenate([p[2].detach().numpy()[:, 0, :] for p in parts]) + 0.5, 0, 1)
    sem = np.concatenate([p[4].detach().numpy().argmax(1) for p in parts])
    op = 1 / (1 + np.exp(-np.concatenate([p[7].detach().numpy()[:, 0] for p in parts])))
    sim = np.stack([xyz[:, 2], -xyz[:, 0], -xyz[:, 1]], 1)
    k = op > 0.5
    return sim[k], col[k], sem[k], op[k]


def main(scene, out, override=None):
    s = np.load(scene + "/pictura_sidecar.npz")
    roads = np.load(override) if override else s["roads"]
    track, agents = s["track"], s["static_agents"]
    P, C, S, O = load(scene)
    lo = track[:, :2].min(0) - MARGIN
    hi = track[:, :2].max(0) + MARGIN
    W, H = (np.ceil((hi - lo) / RES)).astype(int)
    inb = np.all((P[:, :2] >= lo) & (P[:, :2] < hi), 1)
    P, C, S, O = P[inb], C[inb], S[inb], O[inb]
    ix = ((P[:, 0] - lo[0]) / RES).astype(int)
    iy = (H - 1 - (P[:, 1] - lo[1]) / RES).astype(int)
    cell = iy * W + ix

    # ground colour: opacity-weighted mean of road/kerb Gaussians; elsewhere the top-most points, dimmed
    g = np.isin(S, ROAD + KERB + (2,))
    acc = np.zeros((H * W, 3)); w = np.zeros(H * W)
    np.add.at(acc, cell[g], C[g] * O[g, None]); np.add.at(w, cell[g], O[g])
    img = np.zeros((H * W, 3))
    has = w > 0
    img[has] = acc[has] / w[has, None]
    order = np.argsort(P[~g, 2])  # ascending z: later (higher) wins
    cg = cell[~g][order]
    top = np.zeros((H * W, 3)); top[cg] = C[~g][order]
    seen = np.zeros(H * W, bool); seen[cg] = True
    img[~has & seen] = 0.45 * top[~has & seen]
    rgb = Image.fromarray((img.reshape(H, W, 3) * 255).astype(np.uint8))

    # semantic: majority class per cell among ground classes, then vehicles etc. on top
    votes = {c: np.bincount(cell[S == c], minlength=H * W) for c in SEM_COL}
    ground_votes = np.stack([votes[c] for c in ROAD + KERB], 1)
    semimg = np.zeros((H * W, 3), np.uint8)
    gv = ground_votes.sum(1) > 0
    win = np.array(ROAD + KERB)[ground_votes.argmax(1)]
    for c in ROAD + KERB:
        semimg[gv & (win == c)] = SEM_COL[c]
    for c in (8, 2, 13, 14, 15):
        m = votes[c] > 3
        semimg[m] = SEM_COL[c]
    sem = Image.fromarray(semimg.reshape(H, W, 3))

    # metric: drivable area = cells reachable from the drive without crossing an EDGE
    to_px = lambda xy: np.stack([(xy[:, 0] - lo[0]) / RES, H - 1 - (xy[:, 1] - lo[1]) / RES], 1)
    edge = Image.new("L", (W, H), 0); de = ImageDraw.Draw(edge)
    for r in roads[roads[:, 6] == 2]:
        a, b = to_px(r[None, 0:2])[0], to_px(r[None, 3:5])[0]
        de.line([tuple(a), tuple(b)], fill=1, width=2)
    edge = np.array(edge, bool)
    lab, _ = ndimage.label(~edge)
    tp = to_px(track[:, :2]).round().astype(int)
    tp = tp[(tp[:, 0] >= 0) & (tp[:, 0] < W) & (tp[:, 1] >= 0) & (tp[:, 1] < H)]
    ids = np.unique(lab[tp[:, 1], tp[:, 0]]); ids = ids[ids > 0]
    drv = np.isin(lab, ids) | edge
    road_sem = (gv & (win == 0)).reshape(H, W)
    kerb_sem = (gv & (win != 0)).reshape(H, W)
    trk = np.ones((H, W), bool); trk[tp[:, 1], tp[:, 0]] = False
    near = ndimage.distance_transform_edt(trk) * RES < NEAR
    obs = near & (road_sem | kerb_sem)          # cells the Gaussians say something about
    inter = (drv & road_sem & obs).sum()
    m = dict(scene=scene, iou=float(inter / max(((drv | road_sem) & obs).sum(), 1)),
             recall=float(inter / max((road_sem & obs).sum(), 1)),        # semantic road covered by raster
             precision=float(inter / max((drv & obs).sum(), 1)),         # raster road that is semantic road
             raster_on_kerb_m2=float((drv & kerb_sem & near).sum() * RES * RES),
             road_missed_m2=float((~drv & road_sem & near).sum() * RES * RES),
             n_edge=int((roads[:, 6] == 2).sum()), n_line=int((roads[:, 6] == 1).sum()),
             lo=[float(v) for v in lo], res=RES, W=int(W), H=int(H))

    # overlays
    font = ImageFont.load_default()
    for im in (rgb, sem):
        d = ImageDraw.Draw(im)
        for gx in np.arange(np.ceil(lo[0] / 10) * 10, hi[0], 10):
            x = (gx - lo[0]) / RES; d.line([(x, 0), (x, H)], fill=(255, 255, 0) if gx == 0 else (70, 70, 0), width=1)
            d.text((x + 2, 2), f"{gx:.0f}", fill=(255, 255, 0), font=font)
        for gy in np.arange(np.ceil(lo[1] / 10) * 10, hi[1], 10):
            y = H - 1 - (gy - lo[1]) / RES; d.line([(0, y), (W, y)], fill=(255, 255, 0) if gy == 0 else (70, 70, 0), width=1)
            d.text((2, y + 2), f"{gy:.0f}", fill=(255, 255, 0), font=font)
        for cls, colr, wd in ((1, (255, 255, 255), 1), (0, (0, 230, 255), 2), (2, (255, 30, 30), 3)):
            for r in roads[roads[:, 6] == cls]:
                a, b = to_px(r[None, 0:2])[0], to_px(r[None, 3:5])[0]
                d.line([tuple(a), tuple(b)], fill=colr, width=wd)
        for a in agents:
            x, y, _, w_, l, _, yaw = a[:7]
            cs, sn = np.cos(yaw), np.sin(yaw)
            cr = np.array([[l / 2, w_ / 2], [l / 2, -w_ / 2], [-l / 2, -w_ / 2], [-l / 2, w_ / 2]]) @ np.array([[cs, sn], [-sn, cs]]) + [x, y]
            d.polygon([tuple(p) for p in to_px(cr)], outline=(255, 220, 0))
        st = to_px(track[:1, :2])[0]; d.ellipse([st[0] - 6, st[1] - 6, st[0] + 6, st[1] + 6], outline=(0, 255, 0), width=3)
    both = Image.new("RGB", (2 * W + 10, H), (255, 255, 255))
    both.paste(rgb, (0, 0)); both.paste(sem, (W + 10, 0))
    both.save(out + ".png")
    json.dump(m, open(out + ".json", "w"))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))


if __name__ == "__main__":
    main(*sys.argv[1:])
