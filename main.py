#dev/creator:tubakhxn
import argparse
import time
import cv2
import numpy as np
from scene import Scene, Renderer
from camera import CameraSegmenter, demo_mask, demo_hands

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--width', type=int, default=540)
    ap.add_argument('--art-height', type=int, default=540)
    ap.add_argument('--cell', type=int, default=3)
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--no-panel', action='store_true')
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--snapshot', type=str, default='')
    return ap.parse_args()

def main():
    a = parse()
    cell = a.cell
    gw, gh = (a.width // cell, a.art_height // cell)
    pw = gw * cell
    ph = 0 if a.no_panel else int(pw * 0.63)
    scene = Scene(gw, gh)
    rend = Renderer(gw, gh, cell)
    blank = np.zeros((ph, pw, 3), np.uint8)
    cam = None
    if not a.demo and (not a.snapshot):
        cam = CameraSegmenter(gw, gh, pw, max(ph, 1), a.camera, a.invert)
        cam.start()
    win = 'tubakhxn'
    total_h = rend.H + ph
    if not a.snapshot:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, rend.W, total_h)
        if a.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    full = a.fullscreen
    t0 = time.perf_counter()
    n = 0
    try:
        while True:
            t = time.perf_counter() - t0
            if a.snapshot:
                t = n / 30.0
            if cam is None:
                mask, luma, hands, panel = (demo_mask(gw, gh, t), None, demo_hands(gw, gh, t), None)
            else:
                mask, luma, hands, panel = cam.latest()
            art = rend.draw(scene.frame(t, mask, luma, hands))
            n += 1
            if ph:
                img = np.vstack([art, panel if panel is not None else blank])
            else:
                img = art
            if a.snapshot:
                if n >= int(a.snapshot.split(':')[1]):
                    cv2.imwrite(a.snapshot.split(':')[0], img)
                    break
                continue
            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 255
            if k in (27, ord('q')):
                break
            if k == ord('f'):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if k == ord('c') and cam is not None:
                cam.next_camera()
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
