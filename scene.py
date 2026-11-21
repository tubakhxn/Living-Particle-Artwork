#dev/creator:tubakhxn
import math
import numpy as np
import cv2
BLUE_STOPS = [(0.0, (0, 0, 0)), (0.12, (6, 6, 60)), (0.35, (25, 30, 190)), (0.6, (70, 80, 255)), (0.82, (150, 165, 255)), (1.0, (235, 240, 255))]
VIOLET_STOPS = [(0.0, (0, 0, 0)), (0.12, (30, 4, 70)), (0.35, (95, 25, 205)), (0.6, (140, 70, 255)), (0.82, (195, 160, 255)), (1.0, (240, 230, 255))]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

def _make_lut(stops):
    v = np.linspace(0, 1, 256)
    xs = [s[0] for s in stops]
    out = np.zeros((256, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(v, xs, [s[1][c] for s in stops])
    return out[:, ::-1].copy()

def smooth_noise(rng, h, w, ch, cw):
    small = rng.random((ch, cw)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

def wind(x, t):
    gust = 0.75 + 0.25 * math.sin(t * 0.23)
    return gust * (0.65 * np.sin(x * 0.045 - t * 1.15) + 0.35 * np.sin(x * 0.12 - t * 2.4 + 1.3)) + 0.25 * math.sin(t * 0.7)

class Scene:

    def __init__(self, gw, gh, seed=7):
        self.gw, self.gh = (gw, gh)
        self.rng = np.random.default_rng(seed)
        self.ground = int(gh * 0.72)
        self.lut_b = _make_lut(BLUE_STOPS)
        self.lut_d = _make_lut(VIOLET_STOPS) - self.lut_b
        self.vmap = np.clip(smooth_noise(self.rng, gh, gw, 4, 7) * 1.2 - 0.55, 0, 0.3)[..., None]
        self.m_ema = np.zeros((gh, gw), np.float32)
        yy = np.arange(gh, dtype=np.float32)[:, None]
        self.sitfade = self._smooth(np.clip((int(gh * 0.80) - yy) / (0.07 * gh), 0, 1))
        self.rowfade = 1.0 - 0.55 * np.clip((yy - self.ground) / max(1, gh - self.ground), 0, 1)
        self.k = gh / 144.0
        self.hp = [0.0, 0.0]
        self.hpts = [None, None]
        self._build_stars()
        self._build_tree()
        self._build_grass()

    def _build_stars(self):
        r, gw, g = (self.rng, self.gw, self.ground)
        n = int(gw * g * 0.012)
        self.s_x = r.integers(0, gw, n).astype(np.float32)
        self.s_y = (r.random(n) ** 1.2 * (g - 8)).astype(np.float32)
        self.s_b = (0.25 + 0.6 * r.random(n)).astype(np.float32)
        self.s_ph = r.uniform(0, 6.28, n).astype(np.float32)
        self.s_sp = r.uniform(0.6, 3.0, n).astype(np.float32)

    def _leaf(self, cx, cy, rad, n):
        r = self.rng
        a = r.uniform(0, math.pi)
        u = r.normal(0, rad, n)
        v = r.normal(0, rad * 0.4, n)
        x = cx + u * math.cos(a) - v * math.sin(a)
        y = cy + u * math.sin(a) + v * math.cos(a)
        return np.stack([x, y, 0.3 + 0.5 * r.random(n)], 1)

    def _branch(self, x, y, ang, length, depth, width, maxd, acc, leaves):
        r = self.rng
        n = max(2, int(length / 0.6))
        t = np.linspace(0, 1, n)
        bend = r.uniform(-0.18, 0.18) * length
        dx, dy = (math.sin(ang), -math.cos(ang))
        px, py = (-dy, dx)
        cx = x + dx * length * t + px * bend * np.sin(t * math.pi)
        cy = y + dy * length * t + py * bend * np.sin(t * math.pi)
        half = max(0, int(width / 2))
        for k in range(-half, half + 1):
            acc.append(np.stack([cx + px * k, cy + py * k, 0.55 + 0.4 * r.random(n)], 1))
        ex, ey = (cx[-1], cy[-1])
        if depth >= 2 and r.random() < 0.5:
            leaves.append((ex, ey))
        if depth >= maxd or length < 5:
            leaves.append((ex, ey))
            return
        for _ in range(int(r.choice([2, 3], p=[0.55, 0.45]))):
            na = float(np.clip(ang + r.uniform(-0.95, 0.95), -1.9, 1.9))
            self._branch(ex, ey, na, length * r.uniform(0.62, 0.8), depth + 1, width * 0.72, maxd, acc, leaves)

    def _fronds(self, cx, cy, rx, ry, n, parts, lo=4, hi=10):
        r, k = self.rng, self.k
        px = r.normal(cx, rx, n)
        py = r.normal(cy, ry, n)
        ang = np.pi / 2 + r.normal(0, 0.95, n)
        L = r.uniform(lo, hi, n) * k
        t = np.linspace(0, 1, int(hi * k / 0.7) + 1)[None, :]
        xs = px[:, None] + np.cos(ang)[:, None] * L[:, None] * t
        ys = py[:, None] + np.sin(ang)[:, None] * L[:, None] * t + 0.12 * L[:, None] * t * t
        b = 0.35 + 0.6 * r.random(n)[:, None] * (1 - 0.45 * t)
        parts.append(np.stack([xs.ravel(), ys.ravel(), b.ravel()], 1))

    def _build_tree(self):
        r, gw, gh, g = (self.rng, self.gw, self.gh, self.ground)
        tx, th = (0.84 * gw, 0.70 * gh)
        acc, leaves, parts = ([], [], [])
        top = g - th
        for row in range(int(top), g + 3):
            frac = max(0.0, (g - row) / th)
            wd = 0.022 * gw * (1 - 0.45 * frac) + 3 * max(0, 1 - frac * 8) + 1
            cxm = tx + math.sin(frac * 3.0) * 1.5
            xs = np.arange(cxm - wd / 2, cxm + wd / 2 + 0.01, 1.0)
            b = 0.5 + 0.4 * r.random(len(xs))
            b[0] = b[-1] = 0.9
            parts.append(np.stack([xs, np.full_like(xs, row), b], 1))
        specs = [(1.0, -1.3, .30), (0.97, -0.8, .34), (0.95, -0.4, .30), (1.0, 0.5, .26), (0.9, 1.1, .2),
                 (0.82, -1.2, .30), (0.72, 1.25, .2), (0.66, -1.1, .26), (0.55, 1.0, .16), (0.5, -1.2, .2)]
        for fr, ang, l in specs:
            self._branch(tx, g - fr * th, ang, l * th, 0, 2, 4, acc, leaves)
        parts.extend(acc)
        nb = sum(len(p) for p in parts)
        self._fronds(0.50 * gw, 0.17 * gh, 0.20 * gw, 0.10 * gh, 1400, parts)
        self._fronds(0.84 * gw, 0.12 * gh, 0.10 * gw, 0.08 * gh, 250, parts)
        for (lx, ly) in leaves:
            self._fronds(lx, ly, 2.5 * self.k, 2.5 * self.k, 14, parts, 3, 7)
        P = np.concatenate(parts).astype(np.float32)
        leaf = np.zeros(len(P), np.float32)
        leaf[nb:] = 1.0
        ok = (P[:, 0] > -3) & (P[:, 0] < gw + 3) & (P[:, 1] > -3) & (P[:, 1] < g + 4)
        P, leaf = (P[ok], leaf[ok])
        self.t_x, self.t_y, self.t_b, self.t_leaf = (P[:, 0], P[:, 1], P[:, 2], leaf)
        height = np.clip((g - P[:, 1]) / g, 0, 1)
        self.t_w = 1.9 * height ** 1.8 + 0.5 * leaf
        self.t_ph = r.uniform(0, 6.28, len(P)).astype(np.float32)
        self.t_sp = r.uniform(1.5, 4.0, len(P)).astype(np.float32)

    def _build_grass(self):
        r, gw, gh, g = (self.rng, self.gw, self.gh, self.ground)
        n = int(gw * (gh - g) * 0.55)
        bx = r.random(n) * gw
        by = g - 4 + r.random(n) ** 0.85 * (gh - g + 4)
        d = np.clip((by - (g - 4)) / (gh - g + 4), 0, 1)
        h = (4 + r.random(n) * 10) * (0.7 + 0.9 * d) * self.k
        h *= np.where(r.random(n) < 0.08, 1.6, 1.0)
        h = np.maximum(2, h.astype(np.int32))
        noise = smooth_noise(r, gh, gw, 7, 14)
        keep = r.random(n) < np.clip(0.18 + noise[np.clip(by.astype(int), 0, gh - 1), np.clip(bx.astype(int), 0, gw - 1)] * 1.1, 0, 1)
        bx, by, d, h = (bx[keep], by[keep], d[keep], h[keep])
        n = len(bx)
        bl = np.repeat(np.arange(n), h)
        k = np.arange(len(bl)) - np.repeat(np.cumsum(h) - h, h)
        frac = (k / h[bl]).astype(np.float32)
        base = (0.16 + 0.3 * d) * (0.55 + 0.45 * r.random(n))
        self.g_x = bx[bl].astype(np.float32)
        self.g_y = (by[bl] - k).astype(np.float32)
        self.g_frac = frac
        self.g_b = np.clip(base[bl] + 0.34 * frac ** 1.5, 0, 0.72).astype(np.float32)
        amp = (0.8 + 1.4 * r.random(n)) * (0.6 + 0.6 * d)
        self.g_amp = amp[bl].astype(np.float32)
        self.g_ph = r.uniform(0, 6.28, n)[bl].astype(np.float32)
        self.g_fr = d[bl] > 0.55
        self.g_sp = r.uniform(1.0, 3.2, n)[bl].astype(np.float32)

    def _splat(self, F, x, y, v):
        gw, gh = (self.gw, self.gh)
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)
        ok = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
        idx = yi[ok] * gw + xi[ok]
        flat = F.reshape(-1)
        flat[idx] = np.maximum(flat[idx], v[ok])

    @staticmethod
    def _smooth(x):
        return x * x * (3 - 2 * x)

    def _update_hands(self, hands):
        for i in range(2):
            if hands and i < len(hands):
                pts, pr = hands[i]
                self.hpts[i] = pts if self.hpts[i] is None else 0.5 * self.hpts[i] + 0.5 * pts
                self.hp[i] = self.hp[i] * 0.75 + pr * 0.25
            else:
                self.hp[i] *= 0.85

    def _hand_layer(self, t):
        gw, gh, r = (self.gw, self.gh, self.rng)
        H = np.zeros((gh, gw), np.float32)
        for i in range(2):
            s = float(np.clip((self.hp[i] - 0.12) / 0.88, 0, 1))
            pts = self.hpts[i]
            if s <= 0 or pts is None:
                continue
            xs, ys, vs = ([], [], [])
            for a, b in HAND_CONN:
                d = float(np.hypot(*pts[b] - pts[a]))
                n = int(d * 1.6) + 2
                tt = np.linspace(0, 1, n)
                for _ in range(int(2 + 4 * s)):
                    jit = 0.25 + 0.6 * s
                    xs.append(pts[a, 0] + (pts[b, 0] - pts[a, 0]) * tt + r.normal(0, jit, n))
                    ys.append(pts[a, 1] + (pts[b, 1] - pts[a, 1]) * tt + r.normal(0, jit, n))
                    vs.append(0.6 + 0.4 * s * r.random(n))
            c = pts[[0, 5, 9, 13, 17]].mean(0)
            size = float(np.hypot(*pts[9] - pts[0]))
            n = int(150 * s) + 30
            xs.append(c[0] + r.normal(0, size * 0.45, n))
            ys.append(c[1] + r.normal(0, size * 0.45, n))
            vs.append(0.5 + 0.4 * r.random(n))
            for k in (4, 8, 12, 16, 20):
                n = int(30 * s) + 8
                rad = 0.6 + 1.2 * s
                xs.append(pts[k, 0] + r.normal(0, rad, n))
                ys.append(pts[k, 1] + r.normal(0, rad, n))
                vs.append(np.full(n, 0.97))
            n = int(60 * s) + 10
            ang = r.uniform(0, 6.28, n)
            rr = size * (1.5 + 0.2 * np.sin(t * 3 + ang * 3))
            xs.append(c[0] + np.cos(ang) * rr)
            ys.append(c[1] + np.sin(ang) * rr)
            vs.append(np.full(n, 0.45 * s + 0.1))
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.float32)
            v = np.concatenate(vs).astype(np.float32)
            keep = r.random(len(x)) < 0.35 + 0.65 * s
            self._splat(H, x[keep], y[keep], v[keep])
        return H

    def frame(self, t, mask=None, luma=None, hands=None):
        gw, gh, g, r = (self.gw, self.gh, self.ground, self.rng)
        E = np.zeros((gh, gw), np.float32)
        self._update_hands(hands)
        tw = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(t * self.s_sp + self.s_ph)) ** 3
        self._splat(E, self.s_x, self.s_y, self.s_b * tw)
        wt = wind(self.t_x, t)
        dx = wt * 2.2 * self.k * self.t_w + self.t_leaf * 0.7 * self.k * np.sin(t * self.t_sp + self.t_ph)
        dy = self.t_leaf * 0.5 * np.cos(t * self.t_sp * 0.8 + self.t_ph)
        shim = 0.72 + 0.28 * np.sin(t * self.t_sp * 1.7 + self.t_ph * 2)
        wg = wind(self.g_x, t) + 0.5 * np.sin(t * self.g_sp + self.g_ph)
        gdx = wg * self.g_amp * 1.6 * self.k * self.g_frac ** 1.3
        for i in range(2):
            s = float(np.clip((self.hp[i] - 0.12) / 0.88, 0, 1))
            pts = self.hpts[i]
            if s <= 0 or pts is None:
                continue
            c = pts[[0, 5, 9, 13, 17]].mean(0)
            R = float(np.hypot(*pts[9] - pts[0])) * 1.8 + 6
            ddx = self.g_x - c[0]
            gdx += np.sign(ddx) * np.exp(-(ddx * ddx + (self.g_y - c[1]) ** 2) / (2 * R * R)) * s * 5 * self.g_frac
            tdx = self.t_x - c[0]
            dx += np.sign(tdx) * np.exp(-(tdx * tdx + (self.t_y - c[1]) ** 2) / (2 * R * R)) * s * 4 * (0.3 + self.t_w * 0.5)
        self._splat(E, self.t_x + dx, self.t_y + dy, self.t_b * shim)
        gb = self.g_b * (0.85 + 0.15 * np.sin(t * self.g_sp * 2 + self.g_ph))
        self._splat(E, self.g_x + gdx, self.g_y, gb)
        Fl = np.zeros_like(E)
        fr = self.g_fr
        self._splat(Fl, (self.g_x + gdx)[fr], self.g_y[fr], gb[fr])
        P = np.zeros((gh, gw), np.float32)
        if mask is not None:
            if mask.mean() < 0.004:
                self.m_ema *= 0.85
            else:
                self.m_ema = 0.4 * self.m_ema + 0.6 * mask
        m = self.m_ema
        if m.max() > 0.02:
            p = np.clip((m - 0.3) / 0.3, 0, 1)
            p = p * p * (3 - 2 * p)
            p = p * self.sitfade
            rnd = r.random((gh, gw), dtype=np.float32)
            lit = (rnd < p * 0.93) & (r.random((gh, gw), dtype=np.float32) < self.rowfade)
            bright = 0.66 + 0.2 * r.random((gh, gw), dtype=np.float32)
            if luma is not None:
                bright += 0.35 * (luma - 0.45)
            blur = cv2.GaussianBlur(m, (0, 0), 2.0)
            bright += np.clip((m - blur) * 2.5, 0, 0.3)
            P = np.where(lit, np.clip(bright, 0.3, 0.82), 0).astype(np.float32)
            halo = (r.random((gh, gw), dtype=np.float32) < blur * 0.3) & ~lit
            P = np.maximum(P, np.where(halo, 0.38, 0).astype(np.float32))
            P = np.where(Fl > 0, P * 0.3, P)
            E *= 1 - 0.8 * cv2.GaussianBlur(p, (0, 0), 1.0)
        H = self._hand_layer(t)
        if H.max() > 0:
            E *= 1 - 0.6 * cv2.GaussianBlur((H > 0).astype(np.float32), (0, 0), 1.5)
        out = np.maximum(np.maximum(E * 1.35, P), H)
        out *= 0.88 + 0.12 * r.random((gh, gw), dtype=np.float32)
        idx = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        col = self.lut_b[idx] + self.lut_d[idx] * self.vmap
        return np.clip(col, 0, 255).astype(np.uint8)

class Renderer:

    def __init__(self, gw, gh, cell, seed=3):
        self.gw, self.gh, self.c = (gw, gh, cell)
        self.W, self.H = (gw * cell, gh * cell)
        c = cell
        yy, xx = np.mgrid[0:c, 0:c].astype(np.float32)
        d = np.hypot(xx - (c - 1) / 2, yy - (c - 1) / 2)
        m = np.clip((c * 0.42 + 0.7 - d) / 1.4, 0, 1)
        m = (0.05 + 0.95 * m) * 255
        tile = np.tile(m, (gh, gw)).astype(np.uint8)
        self.dots = np.dstack([tile] * 3)
        Y, X = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        rr = np.hypot((X / self.W - 0.5) * 1.1, (Y / self.H - 0.5) * 1.1)
        vig = np.clip(1.15 - rr * 1.1, 0.35, 1.0)
        self.vig = np.dstack([(vig * 255).astype(np.uint8)] * 3)
        rng = np.random.default_rng(seed)
        self.grain = []
        for _ in range(5):
            n = rng.integers(0, 8, (self.H, self.W), dtype=np.uint8)
            self.grain.append(np.dstack([n, (n * 0.45).astype(np.uint8), (n * 0.6).astype(np.uint8)]))
        self.k = 0

    def draw(self, col):
        W, H = (self.W, self.H)
        up = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        out = cv2.multiply(up, self.dots, scale=1 / 255.0)
        b1 = cv2.resize(cv2.GaussianBlur(col, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR)
        b2 = cv2.resize(cv2.GaussianBlur(col, (0, 0), 4.0), (W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(out, 1.0, b1, 0.35, 0)
        out = cv2.addWeighted(out, 1.0, b2, 0.4, 0)
        out = cv2.multiply(out, self.vig, scale=1 / 255.0)
        out = cv2.add(out, self.grain[self.k % len(self.grain)])
        self.k += 1
        return out
