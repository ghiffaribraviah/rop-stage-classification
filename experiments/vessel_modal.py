import cv2
import numpy as np
from scipy import ndimage as ndi

def read_rgb(p):
    i = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(i, cv2.COLOR_BGR2RGB) if i is not None else None

def resize_ms(img, ms):
    h, w = img.shape[:2]
    s = min(1.0, ms / max(h, w))
    return img.copy() if s >= 1.0 else cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA)

def est_fov(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (g > max(3, int(np.percentile(g, 1)))).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if nl > 1:
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m.astype(bool)

def n01(img, mask=None):
    v = img[mask] if mask is not None else img.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0: return np.zeros(img.shape, np.float32)
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo: lo, hi = float(v.min()), float(v.max())
    if hi <= lo: return np.zeros(img.shape, np.float32)
    return np.clip((img.astype(np.float32) - float(lo)) / float(hi-lo), 0, 1).astype(np.float32)

def gk(sz, th, sg, lm):
    c = sz // 2
    y, x = np.ogrid[-c:sz-c, -c:sz-c]
    xt = x * np.cos(th) + y * np.sin(th)
    yt = -x * np.sin(th) + y * np.cos(th)
    gb = np.exp(-0.5 * (xt**2 / sg**2 + yt**2 * 0.25 / sg**2)) * np.cos(2 * np.pi * xt / lm)
    return (gb - gb.mean()).astype(np.float32)

def gabor_resp(inv_f, fov):
    r = np.zeros(inv_f.shape, np.float32)
    for sg, lm in [(1.5, 3), (2.5, 5), (3.5, 7), (5, 10)]:
        sz = max(7, int(6*sg) + (1 - int(6*sg) % 2))
        for a in range(0, 180, 15):
            k = gk(sz, np.deg2rad(a), sg, lm)
            r = np.maximum(r, cv2.filter2D(inv_f, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT))
    r[~fov] = 0
    return n01(r, fov)

def seg_vessels(path):
    rgb = read_rgb(path)
    if rgb is None: return None, None, None
    wrk = resize_ms(rgb, 768)
    fov = est_fov(wrk)
    g = wrk[:, :, 1].copy()
    g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g)
    enh[~fov] = 0
    inv = 255 - enh
    inv_f = n01(inv.astype(np.float32), fov)
    gab = gabor_resp(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7)
    r7[~fov] = 0
    soft = n01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
    u8[~fov] = 0
    enh2 = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12)).apply(u8)
    enh2[~fov] = 0
    sharp = n01(enh2.astype(np.float32), fov)
    inn = erode_fov(fov, 8)
    vals = sharp[inn]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.zeros(fov.shape, bool), np.zeros(fov.shape, np.float32), fov
    th = float(np.percentile(vals, 84))
    bin = (sharp >= th) & inn
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(bin.astype(np.uint8), 8)
    if nl > 1:
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)]
        areas.sort(reverse=True)
        keep = {idx for _, idx in areas[:2]}
        bin = np.isin(labels, list(keep))
    return bin, soft, fov

def erode_fov(mask, r):
    if r <= 0: return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv2.erode(mask.astype(np.uint8), k, 1).astype(bool)
