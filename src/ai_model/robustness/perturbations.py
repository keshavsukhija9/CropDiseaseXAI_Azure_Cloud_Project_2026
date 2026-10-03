import numpy as np
import cv2


def brightness(img, factor=1.4):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def shadow(img, strength=0.5):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    mask[: h // 2, : w // 2] *= (1 - strength)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def haze(img, strength=0.4):
    white = np.full_like(img, 255)
    return cv2.addWeighted(img, 1 - strength, white, strength, 0)


def blur(img, ksize=7):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def jpeg_compress(img, quality=15):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def rotate(img, angle=15):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def occlusion(img, frac=0.2):
    h, w = img.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y0, x0 = np.random.randint(0, h - bh), np.random.randint(0, w - bw)
    out = img.copy()
    out[y0:y0 + bh, x0:x0 + bw] = 0
    return out


def color_shift(img, shift=(20, -10, 10)):
    out = img.astype(np.int16)
    for c in range(3):
        out[..., c] = np.clip(out[..., c] + shift[c], 0, 255)
    return out.astype(np.uint8)


PERTURBATIONS = {
    "brightness": brightness,
    "shadow": shadow,
    "haze": haze,
    "blur": blur,
    "jpeg_compression": jpeg_compress,
    "rotation": rotate,
    "occlusion": occlusion,
    "color_shift": color_shift,
}
