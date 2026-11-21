#dev/creator:tubakhxn
import os
import threading
import time
import urllib.request
import numpy as np
import cv2

BASE = "https://storage.googleapis.com/mediapipe-models/"
SEG_LAND_URL = BASE + "image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
POSE_URL = BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
SEG_URL = BASE + "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
HAND_URL = BASE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DET_URL = BASE + "object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ART_SCALE = 0.85
ART_BOTTOM = 0.80


def fetch(name, url):
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        os.makedirs(MODELS, exist_ok=True)
        print("[info] Downloading %s (one time)..." % name)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def demo_mask(gw, gh, t):
    m = np.zeros((gh, gw), np.uint8)
    cx = int(gw * (0.5 + 0.10 * np.sin(t * 0.4)))
    feet = int(gh * ART_BOTTOM)
    H = int(gh * 0.40)
    cv2.circle(m, (cx, feet - H + int(H * 0.07)), max(2, int(H * 0.075)), 255, -1)
    sh = (cx, feet - int(H * 0.82))
    hip = (cx, feet - int(H * 0.45))
    cv2.line(m, sh, hip, 255, max(3, int(H * 0.2)))
    cv2.line(m, hip, (cx - int(H * .07), feet), 255, max(2, int(H * .09)))
    cv2.line(m, hip, (cx + int(H * .07), feet), 255, max(2, int(H * .09)))
    a = 1.2 + 0.5 * np.sin(t * 3.0)
    hand = (int(sh[0] + np.sin(a) * H * .35), int(sh[1] - np.cos(a) * H * .35))
    cv2.line(m, (sh[0] + int(H * .08), sh[1]), hand, 255, max(2, int(H * .07)))
    cv2.line(m, (sh[0] - int(H * .08), sh[1]), (cx - int(H * .14), hip[1]), 255, max(2, int(H * .07)))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.8)


def demo_hands(gw, gh, t):
    prox = 0.5 + 0.5 * np.sin(t * 0.8)
    s = 6 + 12 * prox
    cx, cy = gw * 0.22, gh * 0.55
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (cx, cy + s * 0.9)
    for base, ang in [(1, -1.0), (5, -0.3), (9, 0.0), (13, 0.25), (17, 0.5)]:
        start = np.array([cx + ang * s * 0.55, cy])
        d = np.array([np.sin(ang * 0.8), -np.cos(ang * 0.8)])
        for j in range(4):
            pts[base + j] = start + d * s * 0.42 * j
    return [(pts, float(prox))]


def draw_box(panel, box, label, crop):
    ph, pw = panel.shape[:2]
    x0c, y0c, cw, ch, w, h = crop
    u0, v0, u1, v1 = box
    X0 = int(np.clip((u0 * w - x0c) / cw * pw, 1, pw - 2))
    X1 = int(np.clip((u1 * w - x0c) / cw * pw, 1, pw - 2))
    Y0 = int(np.clip((v0 * h - y0c) / ch * ph, 1, ph - 2))
    Y1 = int(np.clip((v1 * h - y0c) / ch * ph, 1, ph - 2))
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return
    col = (240, 110, 150)
    cv2.rectangle(panel, (X0, Y0), (X1, Y1), col, 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    ty = max(Y0, th + 12)
    cv2.rectangle(panel, (X0 - 1, ty - th - 12), (X0 + tw + 12, ty), col, -1)
    cv2.putText(panel, label, (X0 + 6, ty - 7), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class CameraSegmenter(threading.Thread):
    def __init__(self, gw, gh, pw, ph, index=0, invert=False):
        super().__init__(daemon=True)
        self.gw, self.gh, self.pw, self.ph, self.index = gw, gh, pw, ph, index
        self.bottom = int(gh * ART_BOTTOM)
        self.lock = threading.Lock()
        self.out = None
        self.active = False
        self.running = True
        self._switch = False
        self.cap = None
        self.seg = None
        self.tseg = None
        self.bg = None
        self.hdet = None
        self.ddet = None
        self.pdet = None
        self.det = None
        self.det_t = 0.0
        self.invert = invert
        self._seen = set()
        self._mp = None
        self._init_models()

    def _init_models(self):
        try:
            import mediapipe as mp
            self._mp = mp
        except Exception as e:
            print("[warn] mediapipe import failed:", e)
        mp = self._mp
        if mp is not None:
            if hasattr(mp, "solutions"):
                try:
                    self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                    print("[info] Selfie segmentation ready.")
                except Exception as e:
                    print("[warn] legacy segmentation failed:", e)
            try:
                from mediapipe.tasks import python as mpt
                from mediapipe.tasks.python import vision
            except Exception as e:
                print("[warn] mediapipe tasks unavailable:", e)
                vision = None
            if vision is not None:
                if self.seg is None:
                    for nm, url in (("selfie_segmenter_landscape.tflite", SEG_LAND_URL), ("selfie_segmenter.tflite", SEG_URL)):
                        try:
                            o = vision.ImageSegmenterOptions(
                                base_options=mpt.BaseOptions(model_asset_path=fetch(nm, url)),
                                running_mode=vision.RunningMode.IMAGE,
                                output_confidence_masks=True, output_category_mask=False)
                            self.tseg = vision.ImageSegmenter.create_from_options(o)
                            print("[info] Segmenter ready (%s)." % nm)
                            break
                        except Exception as e:
                            print("[warn] segmenter %s failed: %s" % (nm, e))
                try:
                    o = vision.ObjectDetectorOptions(
                        base_options=mpt.BaseOptions(model_asset_path=fetch("efficientdet_lite0.tflite", DET_URL)),
                        running_mode=vision.RunningMode.IMAGE, max_results=5, score_threshold=0.25,
                        category_allowlist=["person"])
                    self.ddet = vision.ObjectDetector.create_from_options(o)
                    print("[info] Person detector ready.")
                except Exception as e:
                    print("[warn] person detector unavailable:", e)
                try:
                    o = vision.HandLandmarkerOptions(
                        base_options=mpt.BaseOptions(model_asset_path=fetch("hand_landmarker.task", HAND_URL)),
                        running_mode=vision.RunningMode.IMAGE, num_hands=2)
                    self.hdet = vision.HandLandmarker.create_from_options(o)
                    print("[info] Hand tracking ready.")
                except Exception as e:
                    print("[warn] hand tracking unavailable:", e)
                try:
                    o = vision.PoseLandmarkerOptions(
                        base_options=mpt.BaseOptions(model_asset_path=fetch("pose_landmarker_lite.task", POSE_URL)),
                        running_mode=vision.RunningMode.IMAGE, num_poses=1)
                    self.pdet = vision.PoseLandmarker.create_from_options(o)
                    print("[info] Body tracking ready.")
                except Exception as e:
                    print("[warn] body tracking unavailable:", e)
        if self.seg is None and self.tseg is None:
            print("[warn] Using background-subtraction fallback: step out of frame ~2 seconds at start.")
            self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)

    def _open(self, idx):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for be in backends:
            try:
                cap = cv2.VideoCapture(idx, be)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            for res in ((1280, 720), None):
                if res:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                for _ in range(15):
                    ok, f = cap.read()
                    if ok and f is not None and f.size:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                    time.sleep(0.05)
            cap.release()
        return None

    def _err(self, tag, e):
        if tag not in self._seen:
            self._seen.add(tag)
            print("[warn] %s error: %r" % (tag, e))

    def next_camera(self):
        self._switch = True

    def _do_switch(self):
        self._switch = False
        for step in range(1, 7):
            idx = (self.index + step) % 6
            cap = self._open(idx)
            if cap is not None:
                if self.cap is not None:
                    self.cap.release()
                self.cap, self.index = cap, idx
                print("[info] Using camera %d" % idx)
                return
        print("[info] No other camera found.")

    def stop(self):
        self.running = False

    def latest(self):
        with self.lock:
            if not self.active or self.out is None:
                return None, None, [], None
            m, l, h, p = self.out
            return m.copy(), l.copy(), list(h), p

    def _geom(self, sw, sh):
        wd = int(self.gw * ART_SCALE)
        rows = int(round(wd * sh / sw))
        return wd, rows, (self.gw - wd) // 2, self.bottom - rows

    def _place(self, arr, sw, sh):
        wd, rows, x0, y0 = self._geom(sw, sh)
        a = cv2.resize(arr, (wd, rows), interpolation=cv2.INTER_CUBIC)
        out = np.zeros((self.gh, self.gw), np.float32)
        sy0, dy0 = max(0, -y0), max(0, y0)
        n = min(rows - sy0, self.gh - dy0)
        if n > 0:
            out[dy0:dy0 + n, x0:x0 + wd] = a[sy0:sy0 + n]
        return out

    def _segment(self, small):
        m = np.asarray(self._segment_raw(small), np.float32)
        return m[..., 0] if m.ndim == 3 else m

    def _segment_raw(self, small):
        if self.seg is not None:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            return self.seg.process(rgb).segmentation_mask.astype(np.float32)
        if self.tseg is not None:
            rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            masks = self.tseg.segment(img).confidence_masks
            m = np.array(masks[1 if len(masks) > 1 else 0].numpy_view(), dtype=np.float32)
            return 1.0 - m if self.invert else m
        fg = self.bg.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8))
        return cv2.GaussianBlur(fg.astype(np.float32) / 255, (0, 0), 2.0)

    def _rgb480(self, frame):
        h, w = frame.shape[:2]
        hs = cv2.resize(frame, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(cv2.cvtColor(hs, cv2.COLOR_BGR2RGB))

    def _hands(self, rgb, sw, sh):
        if self.hdet is None:
            return []
        res = self.hdet.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        wd, rows, x0, y0 = self._geom(sw, sh)
        hands = []
        for lm in res.hand_landmarks:
            p = np.array([(l.x, l.y) for l in lm], np.float32)
            size = np.hypot((p[9, 0] - p[0, 0]) * sw, (p[9, 1] - p[0, 1]) * sh) / sw
            prox = 0.35 + 0.65 * float(np.clip((size - 0.04) / 0.10, 0, 1))
            g = np.stack([x0 + p[:, 0] * wd, y0 + p[:, 1] * rows], 1).astype(np.float32)
            hands.append((g, prox))
        hands.sort(key=lambda a: a[0][9, 0])
        return hands

    def _pose_mask(self, rgb, sw, sh):
        out = np.zeros((self.gh, self.gw), np.float32)
        if self.pdet is None:
            return out
        res = self.pdet.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return out
        lm = res.pose_landmarks[0]
        wd, rows, x0, y0 = self._geom(sw, sh)

        def vis(i):
            v = getattr(lm[i], "visibility", None)
            return 1.0 if v is None else v

        def pt(i):
            return (int(x0 + lm[i].x * wd), int(y0 + lm[i].y * rows))

        canvas = np.zeros((self.gh, self.gw), np.uint8)
        th = max(2, int(wd * 0.045))
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
            if vis(a) > 0.5 and vis(b) > 0.5:
                cv2.line(canvas, pt(a), pt(b), 255, th)
        for i in (15, 16):
            if vis(i) > 0.5:
                cv2.circle(canvas, pt(i), th, 255, -1)
        return cv2.GaussianBlur(canvas.astype(np.float32) / 255, (0, 0), 0.7)

    def _person_box(self, rgb):
        res = self.ddet.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        hh, ww = rgb.shape[:2]
        best = None
        for d in res.detections:
            c = d.categories[0]
            if c.category_name != "person":
                continue
            b = d.bounding_box
            cand = (b.origin_x / ww, b.origin_y / hh, (b.origin_x + b.width) / ww, (b.origin_y + b.height) / hh, c.score)
            if best is None or cand[4] > best[4]:
                best = cand
        return best

    def _update_box(self, rgb, m, now):
        if self.ddet is not None:
            if self.n % 3 == 0:
                b = self._person_box(rgb)
                if b is not None:
                    if self.det is not None and now - self.det_t < 0.8:
                        b = tuple(0.5 * o + 0.5 * n for o, n in zip(self.det, b))
                    self.det, self.det_t = b, now
            if self.det is not None and now - self.det_t < 0.7:
                return self.det
            return None
        fg = m > 0.5
        if fg.sum() < 60:
            return None
        cols = np.where(fg.sum(0) > 1)[0]
        rws = np.where(fg.sum(1) > 1)[0]
        if not len(cols) or not len(rws):
            return None
        sh, sw = m.shape
        return (cols[0] / sw, rws[0] / sh, (cols[-1] + 1) / sw, (rws[-1] + 1) / sh, float(m[fg].mean()))

    def _panel(self, frame, m, box):
        h, w = frame.shape[:2]
        asp = self.pw / self.ph
        if w / h > asp:
            cw, ch = h * asp, float(h)
        else:
            cw, ch = float(w), w / asp
        x0, y0 = (w - cw) / 2, (h - ch) / 2
        crop = frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
        panel = cv2.resize(crop, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        mh, mw = m.shape
        mc = m[int(y0 / h * mh):max(int((y0 + ch) / h * mh), 1), int(x0 / w * mw):max(int((x0 + cw) / w * mw), 1)]
        mk = cv2.resize(mc, (self.pw, self.ph), interpolation=cv2.INTER_LINEAR)
        fg = mk > 0.5
        if fg.sum() > 400:
            panel[fg] = (panel[fg] * 0.62 + np.array([235, 95, 140]) * 0.38).astype(np.uint8)
        if box is not None:
            draw_box(panel, box[:4], "person:  %.3f" % box[4], (x0, y0, cw, ch, w, h))
        return panel

    def run(self):
        fails = 0
        self.n = 0
        while self.running:
            if self.cap is None:
                for idx in [self.index] + [i for i in range(5) if i != self.index]:
                    cap = self._open(idx)
                    if cap is not None:
                        self.cap, self.index = cap, idx
                        print("[info] Camera %d opened." % idx)
                        break
                if self.cap is None:
                    with self.lock:
                        self.active = False
                    print("[warn] No camera found, retrying... (close other apps using the webcam)")
                    time.sleep(2.0)
                    continue
            if self._switch:
                self._do_switch()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                with self.lock:
                    self.active = False
                if fails > 30:
                    self.cap.release()
                    self.cap = None
                    fails = 0
                time.sleep(0.03)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            sw = 320
            sh = int(sw * h / w)
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            m = np.zeros((sh, sw), np.float32)
            hands, box, rgb, pm = [], None, None, None
            try:
                m = self._segment(small)
            except Exception as e:
                self._err("segmentation", e)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            try:
                rgb = self._rgb480(frame)
                box = self._update_box(rgb, m, time.time())
                hands = self._hands(rgb, sw, sh)
                pm = self._pose_mask(rgb, sw, sh)
            except Exception as e:
                self._err("detection", e)
            try:
                panel = self._panel(frame, m, box)
            except Exception as e:
                self._err("panel", e)
                panel = cv2.resize(frame, (self.pw, self.ph))
            self.n += 1
            try:
                pl = self._place(m, sw, sh)
                if pm is not None:
                    pl = np.maximum(pl, pm)
                out = (pl, self._place(gray, sw, sh), hands, panel)
            except Exception as e:
                self._err("placement", e)
                out = (np.zeros((self.gh, self.gw), np.float32), np.zeros((self.gh, self.gw), np.float32), [], panel)
            with self.lock:
                self.out, self.active = out, True
        if self.cap is not None:
            self.cap.release()
