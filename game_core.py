"""
Ядро игры: шум, рельеф, текстуры, модели, шрифт, матрицы, игровая логика. Без OpenGL."""
import math

import numpy as np

W_SIZE = (1280, 720)
FOV = 75.0
CHUNK = 32
WORLD_CHUNKS = 8
WORLD = CHUNK * WORLD_CHUNKS
VIEW_CHUNKS = 3
FOG_NEAR, FOG_FAR = 55.0, 90.0
WATER_LEVEL = 4.0
DAY_LEN = 150.0
EYE = 1.62
GRAV = 26.0
JUMP_V = 8.2
WALK, SPRINT = 4.6, 7.4
SWIM_SPEED = 2.4
REACH = 2.7
PLAYER_R = 0.38
TILE = 64
ATLAS_COLS, ATLAS_ROWS = 4, 3
T_GRASS, T_ROCK, T_SAND, T_BARK, T_LEAF, T_FLAME, T_FUR, T_WATER, T_RED = range(9)

RECIPES = [
    ("ТОПОР", [("дерево", 3), ("камень", 2)]),
    ("КОПЬЁ", [("дерево", 2), ("камень", 1)]),
    ("КОСТЁР", [("дерево", 5), ("камень", 4)]),
]

UI_STRINGS = (
    "ЗДОРОВЬЕ ГОЛОД ЖАЖДА ТЕПЛО ДЕРЕВО КАМЕНЬ ЯГОДЫ СЫРОЕ МЯСО ЖАРЕНОЕ МЯСО",
    "ДЕНЬ НОЧЬ ПАУЗА КРАФТ: ИНВЕНТАРЬ (TAB) ПЛОТ",
    "1 - ТОПОР: 3 ДЕРЕВА + 2 КАМНЯ",
    "2 - КОПЬЁ: 2 ДЕРЕВА + 1 КАМЕНЬ",
    "3 - КОСТЁР: 5 ДЕРЕВА + 4 КАМНЯ",
    "УЖЕ ЕСТЬ! НЕ ХВАТАЕТ:",
    "КОСТЁР ГОРИТ! ТЕПЛО У ОГНЯ. E У ОГНЯ - ПРИГОТОВИТЬ МЯСО",
    "+1 ДЕРЕВО +3 ДЕРЕВА +1 КАМЕНЬ +2 КАМНЯ +2 ЯГОДЫ +4 ЯГОДЫ",
    "ВЫ ПЬЁТЕ ВОДУ. ЖАЖДА УТЛЕНА",
    "СЪЕДЕНО ЖАРЕНОЕ МЯСО +45 СЪЕДЕНЫ ЯГОДЫ +12 СЫРОЕ МЯСО +8 НЕТ ЕДЫ",
    "МЯСО ПРИГОТОВЛЕНО! ЖМИ F ЧТОБЫ СЪЕСТЬ НЕТ СЫРОГО МЯСА. ОХОТЬСЯ ЛКМ",
    "ВЗЯТО СЫРОЕ МЯСО ВОЛК ПОВЕРЖЕН! +2 СЫРОГО МЯСА КАБАН УБИТ! +3 СЫРОГО МЯСА",
    "ГОЛОД! ЖАЖДА! ХОЛОД! РАЗВЕДИ КОСТЁР (C) ПЛЫВИ (W + ПРОБЕЛ - ВВЕРХ)",
    "НОЧЬ. ВОЛКИ ГОЛОДНЫ. ЖГИ КОСТЁР! КОСТЁР ПОГАС. НУЖЕН НОВЫЙ (C)",
    "ВЫ ПОГИБЛИ ВЫЖИЛ ДНЕЙ: ПРОБЕЛ - ЗАНОВО",
    "ЛЕС. ВЫЖИВАНИЕ WASD - ДВИЖЕНИЕ МЫШЬ - ОБЗОР SHIFT - БЕГ SPACE - ПРЫЖОК",
    "E - ДОБЫТЬ / ВЗЯТЬ / ПИТЬ ВОДУ ЛКМ - АТАКА (КОПЬЁ БЬЁТ СИЛЬНЕЕ)",
    "F - ЕСТЬ C - КРАФТ H - ПОМОЩЬ ESC - ПАУЗА TAB - ИНВЕНТАРЬ",
    "ЗАДАЧА: ВЕЧЕРОМ РАЗВЕДИ КОСТЁР. НОЧЬЮ - ВОЛКИ. ПЕЙ У ВОДЫ, ЕШЬ ЯГОДЫ.",
    "ПРОБЕЛ - НАЧАТЬ ДЕНЬ ПЕРВЫЙ ЗАДАН СИД МИРА: ",
)


HELP_LINES = (
    "WASD - ДВИЖЕНИЕ, МЫШЬ - ОБЗОР, SHIFT - БЕГ, SPACE - ПРЫЖОК / ВСПЛЫТЬ",
    "E - ДОБЫТЬ / ВЗЯТЬ / ПИТЬ ВОДУ У БЕРЕГА, ЛКМ - ДЕЙСТВИЕ СЛОТА",
    "1-4 - СЛОТ (ТОПОР, КОПЬЁ, ЕДА), C - КРАФТ, TAB - ИНВЕНТАРЬ",
    "ЗАДАЧА: НАРУБИ ДЕРЕВЬЕВ И КАМНЯ, ВЕЧЕРОМ - КОСТЁР.",
    "НОЧЬЮ ПРИХОДЯТ ВОЛКИ. ДНЁМ МОЖНО ОХОТИТЬСЯ НА КАБАНОВ.",
)


def _hash(ix: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = ix.astype(np.int64) * 0x27D4EB2D ^ iz.astype(np.int64) * 0x165667B1 ^ int(seed) * 0x9E3779B1
    h ^= h >> np.int64(13)
    h = h * np.int64(0x5BD1E995)
    h ^= h >> np.int64(15)
    return ((h >> np.int64(6)) & 0xFFFF).astype(np.float64) / 65535.0


def vnoise(x: np.ndarray, z: np.ndarray, freq: float, seed: int) -> np.ndarray:
    fx, fz = x * freq, z * freq
    ix, iz = np.floor(fx), np.floor(fz)
    tx, tz = fx - ix, fz - iz
    sx = tx * tx * (3.0 - 2.0 * tx)
    sz = tz * tz * (3.0 - 2.0 * tz)
    ix, iz = ix.astype(np.int64), iz.astype(np.int64)
    h00 = _hash(ix, iz, seed)
    h10 = _hash(ix + 1, iz, seed)
    h01 = _hash(ix, iz + 1, seed)
    h11 = _hash(ix + 1, iz + 1, seed)
    a = h00 + (h10 - h00) * sx
    b = h01 + (h11 - h01) * sx
    return a + (b - a) * sz


def fbm(x: np.ndarray, z: np.ndarray, freq: float, octaves: int, seed: int) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float64)
    amp, norm, f = 1.0, 0.0, freq
    for o in range(octaves):
        total += vnoise(x, z, f, seed + o * 101) * amp
        norm += amp
        amp *= 0.5
        f *= 2.0
    return total / norm


class World:
    def __init__(self, seed: int):
        self.seed = int(seed)
        n = WORLD + 1
        gx, gz = np.meshgrid(np.arange(n, dtype=np.float64),
                             np.arange(n, dtype=np.float64), indexing="ij")
        s = self.seed
        base = fbm(gx, gz, 1.0 / 67.0, 4, s)
        hills = fbm(gx, gz, 1.0 / 23.0, 3, s + 7)
        ridge_n = fbm(gx, gz, 1.0 / 41.0, 3, s + 17)
        ridge = (1.0 - np.abs(2.0 * ridge_n - 1.0)) ** 3.0
        mountain_mask = np.clip((base - 0.58) * 5.0, 0.0, 1.0)
        self.h = ((base - 0.34) * 38.0
                  + (hills - 0.5) * 11.0
                  + ridge * mountain_mask * 22.0)
        self.h = np.clip(self.h, 0.0, 38.0)
        self.forest = fbm(gx, gz, 1.0 / 29.0, 3, s + 13)
        self.tint = fbm(gx, gz, 1.0 / 9.0, 2, s + 29)
        dhdx, dhdz = np.gradient(self.h)
        nl = np.sqrt(1.0 + dhdx * dhdx + dhdz * dhdz)
        self.nx, self.ny, self.nz = -dhdx / nl, 1.0 / nl, -dhdz / nl
        self.max_h = float(self.h.max())

    def height(self, x: float, z: float) -> float:
        x = float(np.clip(x, 0.0, WORLD - 1e-3))
        z = float(np.clip(z, 0.0, WORLD - 1e-3))
        ix, iz = int(x), int(z)
        tx, tz = x - ix, z - iz
        h00, h10 = self.h[ix, iz], self.h[ix + 1, iz]
        h01, h11 = self.h[ix, iz + 1], self.h[ix + 1, iz + 1]
        return float((h00 * (1 - tx) + h10 * tx) * (1 - tz)
                     + (h01 * (1 - tx) + h11 * tx) * tz)


def ground_splat(h: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Веса (трава, камень, песок) для каждого пикселя рельефа."""
    sand_t = np.clip((WATER_LEVEL + 1.7 - h) / 1.2, 0.0, 1.0)
    rock_t = np.clip((0.90 - ny) * 9.0, 0.0, 1.0)
    snow = np.clip((h - 24.0) / 5.0, 0.0, 1.0)
    rock_t = np.clip(rock_t + snow * 0.8, 0.0, 1.0)
    grass = 1.0 - sand_t - rock_t
    grass = np.clip(grass, 0.0, 1.0)
    total = grass + rock_t + sand_t
    return np.stack([grass / total, rock_t / total, sand_t / total], axis=-1)


# ------------------------------------------------------------------ текстуры

def _tile_pixels(size: int = TILE):
    gy, gx = np.mgrid[0:size, 0:size]
    return gx.astype(np.float64), gy.astype(np.float64)


def _clamp01(a):
    return np.clip(a, 0.0, 1.0)


def build_texture_atlas() -> np.ndarray:
    """Атлас 4x3 тайла 64px, RGB + alpha, uint8. Вызывается один раз."""
    size, aw, ah = TILE, ATLAS_COLS, ATLAS_ROWS
    out = np.zeros((ah * size, aw * size, 4), dtype=np.uint8)
    gx, gy = _tile_pixels(size)

    def put(tile_id, rgb, alpha=255):
        r, c = divmod(tile_id, aw)
        out[r * size:(r + 1) * size, c * size:(c + 1) * size, 0] = (rgb[..., 0] * 255).astype(np.uint8)
        out[r * size:(r + 1) * size, c * size:(c + 1) * size, 1] = (rgb[..., 1] * 255).astype(np.uint8)
        out[r * size:(r + 1) * size, c * size:(c + 1) * size, 2] = (rgb[..., 2] * 255).astype(np.uint8)
        out[r * size:(r + 1) * size, c * size:(c + 1) * size, 3] = alpha

    n1 = fbm(gx, gy, 1.0 / 9.0, 4, 101)
    n2 = fbm(gx, gy, 1.0 / 3.5, 2, 202)
    grass = np.stack([0.22 + 0.14 * n1 - 0.05 * n2,
                      0.40 + 0.18 * n1 + 0.06 * n2,
                      0.14 + 0.10 * n1 - 0.04 * n2], axis=-1)
    blade = vnoise(gx * 2.0, gy * 0.6, 1.0 / 2.0, 303)
    grass[..., 0] -= 0.05 * blade
    grass[..., 1] += 0.06 * blade
    put(T_GRASS, _clamp01(grass))

    nr = fbm(gx, gy, 1.0 / 7.0, 4, 404)
    cracks = np.abs(fbm(gx, gy, 1.0 / 2.6, 2, 505) - 0.5)
    rock_v = 0.42 + 0.22 * nr - 0.35 * (cracks < 0.035).astype(np.float64)
    rock = np.stack([rock_v, rock_v * 0.98, rock_v * 0.95], axis=-1)
    put(T_ROCK, _clamp01(rock))

    nsd = fbm(gx, gy, 1.0 / 5.0, 3, 606)
    specks = vnoise(gx * 3.0, gy * 3.0, 1.0, 707)
    sand = np.stack([0.74 + 0.08 * nsd + 0.05 * specks,
                     0.68 + 0.08 * nsd,
                     0.48 + 0.08 * nsd], axis=-1)
    put(T_SAND, _clamp01(sand))

    stretch_dy = gy * 3.0
    nbark = fbm(gx, stretch_dy, 1.0 / 6.0, 4, 808)
    bark = np.stack([0.30 + 0.14 * nbark, 0.21 + 0.10 * nbark, 0.12 + 0.07 * nbark], axis=-1)
    groove = (np.abs(fbm(gx, gy * 4.0, 1.0 / 2.0, 2, 909) - 0.5) < 0.04).astype(np.float64)
    bark -= 0.10 * groove[..., None]
    put(T_BARK, _clamp01(bark))

    nleaf = fbm(gx, gy, 1.0 / 6.0, 4, 111)
    leaf = np.stack([0.10 + 0.10 * nleaf, 0.34 + 0.20 * nleaf, 0.11 + 0.08 * nleaf], axis=-1)
    hole_n = fbm(gx, gy, 1.0 / 4.0, 3, 121)
    gap = hole_n < 0.30
    edge_soft = _clamp01((hole_n - 0.26) * 8.0)
    leaf_alpha = np.where(gap, 0, 1) * (0.4 + 0.6 * edge_soft)
    put(T_LEAF, _clamp01(leaf), alpha=(leaf_alpha * 255).astype(np.uint8))

    dx = (gx - size / 2) / (size / 2)
    dy = (gy - size / 2) / (size / 2)
    rr2 = np.clip(dx * dx + dy * dy, 0.0, 1.0)
    core = np.clip(1.0 - rr2 * 1.9, 0.0, 1.0)
    flick = fbm(gx * 2.0, gy - gx * 0.5, 1.0 / 3.0, 2, 131)
    flame = np.stack([0.78 + 0.20 * core, 0.30 + 0.42 * core + 0.18 * flick, 0.05 + 0.30 * core * core], axis=-1)
    flame_alpha = _clamp01(core * 1.5 + 0.42 * flick - 0.34)
    put(T_FLAME, _clamp01(flame), alpha=(flame_alpha * 255).astype(np.uint8))

    nfur = fbm(gx, gy, 1.0 / 4.0, 4, 141)
    fur = np.stack([0.30 + 0.14 * nfur, 0.27 + 0.12 * nfur, 0.25 + 0.10 * nfur], axis=-1)
    guard = (vnoise(gx * 2.0, gy * 2.0, 1.0 / 2.0, 151) > 0.86).astype(np.float64)
    fur += 0.12 * guard[..., None]
    put(T_FUR, _clamp01(fur))

    nw = fbm(gx + 0.0, gy, 1.0 / 5.0, 3, 161)
    sparkle = vnoise(gx * 2.0, gy * 2.0, 1.0 / 2.0, 171)
    water = np.stack([0.10 + 0.08 * nw + 0.10 * sparkle,
                      0.30 + 0.14 * nw,
                      0.50 + 0.16 * nw + 0.08 * sparkle], axis=-1)
    put(T_WATER, _clamp01(water))

    nred = fbm(gx, gy, 1.0 / 5.0, 3, 181)
    grain = (vnoise(gx * 2.5, gy * 2.5, 1.0 / 2.0, 191) > 0.8).astype(np.float64)
    meat = np.stack([0.62 + 0.18 * nred + 0.10 * grain,
                     0.16 + 0.08 * nred,
                     0.13 + 0.05 * nred], axis=-1)
    put(T_RED, _clamp01(meat))
    return out


def atlas_tile_uv(tile_id: int) -> tuple[float, float]:
    r, c = divmod(tile_id, ATLAS_COLS)
    return c / ATLAS_COLS, r / ATLAS_ROWS


# ------------------------------------------------------------------ примитивы и модели
_ICO_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _smooth_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                 verts[faces[:, 2]] - verts[faces[:, 0]])
    acc = np.zeros_like(verts)
    for i in range(3):
        np.add.at(acc, faces[:, i], n)
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return acc / ln


def _subdivide(v: np.ndarray, f: np.ndarray):
    cache: dict[tuple[int, int], int] = {}
    verts = [tuple(p) for p in v]
    out = np.empty((len(f) * 4, 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(f):
        ids = []
        for u, w in ((a, b), (b, c), (c, a)):
            k = (min(u, w), max(u, w))
            if k not in cache:
                cache[k] = len(verts)
                verts.append(tuple((v[u] + v[w]) / 2.0))
            ids.append(cache[k])
        ab, bc, ca = ids
        out[i * 4] = (a, ab, ca)
        out[i * 4 + 1] = (b, bc, ab)
        out[i * 4 + 2] = (c, ca, bc)
        out[i * 4 + 3] = (ab, bc, ca)
    return np.array(verts), out


def _icosphere(subdiv: int, r: float, squash_y: float = 1.0):
    key = (subdiv, round(r, 4), round(squash_y, 4))
    if key in _ICO_CACHE:
        return _ICO_CACHE[key]
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    for _ in range(subdiv):
        v, f = _subdivide(v, f)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v * r * np.array([1.0, squash_y, 1.0])
    res = (v.copy(), f)
    if len(_ICO_CACHE) < 64:
        _ICO_CACHE[key] = res
    return res


def _displace(verts: np.ndarray, faces: np.ndarray, amp: float, freq: float, seed: int):
    rad = np.linalg.norm(verts, axis=1)
    rad[rad == 0] = 1.0
    n = fbm(verts[:, 0] * freq + 10.0, verts[:, 2] * freq + 10.0, 1.0, 1, seed)
    verts = verts + (verts / rad[:, None]) * ((n - 0.5) * 2.0 * amp)[:, None]
    return verts, _smooth_normals(verts, faces)


def _sphere_uv(verts: np.ndarray) -> np.ndarray:
    u = np.arctan2(verts[:, 2], verts[:, 0]) / (2.0 * np.pi) + 0.5
    vmax = max(np.abs(verts[:, 1]).max(), 1e-6)
    vv = verts[:, 1] / vmax * 0.5 + 0.5
    return np.stack([u, vv], axis=-1)


def _mk9(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, tile: int) -> np.ndarray:
    out = np.empty((len(pos), 9), dtype=np.float32)
    out[:, 0:3] = pos
    out[:, 3:6] = nrm
    out[:, 6:8] = uv
    out[:, 8] = tile
    return out


def _xform9(v9: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = v9.copy()
    out[:, 0:3] = out[:, 0:3] @ m[:, 0:3].T + m[:, 3]
    out[:, 3:6] = out[:, 3:6] @ m[:, 0:3].T
    ln = np.linalg.norm(out[:, 3:6], axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    out[:, 3:6] /= ln
    return out


def _merge9(*parts) -> tuple[np.ndarray, np.ndarray]:
    verts = np.concatenate([p[0] for p in parts], axis=0)
    offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    idx = np.concatenate([p[1] + o for p, o in zip(parts, offs)], axis=0)
    return verts.astype(np.float32), idx.astype(np.uint32)


def _sphere_part(r: float, squash_y: float, cx: float, cy: float, cz: float,
                 tile: int, subdiv: int = 1, amp: float = 0.0,
                 freq: float = 3.0, seed: int = 1,
                 sx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    v, f = _icosphere(subdiv, r, squash_y)
    v[:, 0] *= sx
    if amp > 0.0:
        v, nv = _displace(v, f, amp, freq, seed)
    else:
        nv = _smooth_normals(v, f)
    v = v + np.array([cx, cy, cz])
    return _mk9(v, nv, _sphere_uv(v), tile), np.asarray(f, dtype=np.uint32)


def _tube(r0: float, r1: float, h: float, radial: int = 7, rings: int = 4,
          bend: float = 0.0, tile: int = 3, cy: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
    cx, sx = np.cos(ang), np.sin(ang)
    ys = np.linspace(0.0, h, rings + 1)
    pos, nrm, uv, idx = [], [], [], []
    for iy, y in enumerate(ys):
        t = y / h
        r = r0 + (r1 - r0) * t
        bx = bend * t * t
        for ix in range(radial):
            px, pz = cx[ix] * r + bx, sx[ix] * r
            pos.append((px, y, pz))
            n = np.array([cx[ix], 0.15 * (r0 - r1) / max(r, 1e-6), sx[ix]])
            n /= max(np.linalg.norm(n), 1e-6)
            nrm.append(tuple(n))
            uv.append((ix / radial, t))
    for iy in range(rings):
        for ix in range(radial):
            a = iy * radial + ix
            b = iy * radial + (ix + 1) % radial
            c = a + radial
            d = b + radial
            idx.extend([a, c, b, b, c, d])
    pos = np.array(pos) + np.array([0.0, cy, 0.0])
    return _mk9(np.array(pos), np.array(nrm), np.array(uv), tile), np.asarray(idx, dtype=np.uint32).reshape(-1, 3)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0]])


def _move(dx: float, dy: float, dz: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, dx], [0.0, 1.0, 0.0, dy], [0.0, 0.0, 1.0, dz]])


def _mul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[:, 0:3] = a[:, 0:3] @ b[:, 0:3]
    out[:, 3] = a[:, 0:3] @ b[:, 3] + a[:, 3]
    return out


def tree_model(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if ("tree", seed % 3) in _MODEL_CACHE:
        return _MODEL_CACHE[("tree", seed % 3)]
    seed = seed % 3
    ang = float(fbm(np.array([seed * 1.7]), np.array([seed * 2.3]), 1.0, 1, seed + 3)[0]) * 4.0
    parts = [_tube(0.24, 0.10, 3.4, radial=7, rings=5, bend=0.16 * (1.0 if seed % 2 else -1.0), tile=T_BARK)]
    n_lobe = 4 + seed % 3
    for i in range(n_lobe):
        a = ang + i * (2.0 * np.pi / n_lobe)
        rr = 0.55 + 0.35 * ((seed + i * 7) % 5) / 5.0
        r = 0.95 + 0.3 * ((seed + i * 11) % 4) / 4.0
        ly = 3.1 + 0.9 * (i % 3) + 0.2 * (i % 2)
        parts.append(_sphere_part(r, 0.8, np.cos(a) * rr * 0.8, ly, np.sin(a) * rr * 0.8,
                                  T_LEAF, subdiv=1, amp=0.16, freq=2.5, seed=seed + i * 3, sx=1.15))
    top = _sphere_part(1.05, 0.85, 0.16, 4.6, 0.0, T_LEAF, subdiv=1, amp=0.15, freq=2.5, seed=seed + 40, sx=1.1)
    parts.append(top)
    res = _merge9(*parts)
    _MODEL_CACHE[("tree", seed)] = res
    return res


_MODEL_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _cached(key, fn):
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = fn()
    return _MODEL_CACHE[key]


def rock_model(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    def build():
        v, f = _icosphere(2, 0.7)
        v[:, 1] *= 0.72
        v, nv = _displace(v, f, 0.2, 2.2, 77)
        v[:, 1] -= v[:, 1].min() - 0.05
        return _merge9((_mk9(v, nv, _sphere_uv(v), T_ROCK), f.astype(np.uint32)))
    return _cached("rock", build)


def bush_model(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    def build():
        parts = [_sphere_part(0.42, 0.7, 0.0, 0.34, 0.0, T_LEAF, subdiv=1, amp=0.1, freq=4.0, seed=seed, sx=1.2)]
        for i in range(3):
            a = i * 2.1 + seed
            parts.append(_sphere_part(0.075, 1.0, np.cos(a) * 0.3, 0.34 + 0.12 * (i % 2),
                                      np.sin(a) * 0.3, T_RED, subdiv=0))
        return _merge9(*parts)
    return _cached(("bush", seed % 3), build)


def _leg_part(r0: float, r1: float, length: float, tile: int) -> tuple[np.ndarray, np.ndarray]:
    v, i = _tube(r0, r1, length, radial=6, rings=3, tile=tile)
    v[:, 1] *= -1.0
    v = np.concatenate([v[:, 0:3], -v[:, 3:6], v[:, 6:9]], axis=1)
    return v.astype(np.float32), i


def wolf_model() -> tuple[np.ndarray, np.ndarray, list, list]:
    def build():
        body = _sphere_part(1.0, 1.0, 0.0, 0.0, 0.0, T_FUR)
        body = (_xform9(body[0], np.array([[0.52, 0, 0, 0], [0, 0.44, 0, 0.62], [0, 0, 0.8, 0.12]])), body[1])
        snout = _tube(0.13, 0.07, 0.26, radial=6, rings=2, tile=T_FUR)
        snout = (_xform9(snout[0], _mul3(_move(0.0, 0.7, 0.78), _rot_x(np.pi / 2.0 - 0.25))), snout[1])
        ear_l = _sphere_part(0.09, 1.4, 0.13, 1.02, 0.6, T_FUR, subdiv=0)
        ear_r = _sphere_part(0.09, 1.4, -0.13, 1.02, 0.6, T_FUR, subdiv=0)
        tail = _sphere_part(0.14, 1.0, 0.0, 0.66, -0.82, T_FUR, subdiv=0, sx=0.7)
        head = _sphere_part(0.24, 0.85, 0.0, 0.72, 0.6, T_FUR, subdiv=1, sx=1.1)
        body_v, body_i = _merge9(body, head, snout, ear_l, ear_r, tail)
        hips = [(0.26, 0.6, 0.34), (-0.26, 0.6, 0.34), (0.26, 0.6, -0.36), (-0.26, 0.6, -0.36)]
        legs = [_leg_part(0.075, 0.05, 0.6, T_FUR) for _ in range(4)]
        return body_v, body_i, hips, legs
    return _cached("wolf", build)


def boar_model() -> tuple[np.ndarray, np.ndarray, list, list]:
    def build():
        body = _sphere_part(1.0, 1.0, 0.0, 0.0, 0.0, T_BARK)
        body = (_xform9(body[0], np.array([[0.55, 0, 0, 0], [0, 0.46, 0, 0.5], [0, 0, 0.78, 0.0]])), body[1])
        snout = _sphere_part(0.14, 0.8, 0.0, 0.5, 0.85, T_SAND, subdiv=0)
        body_v, body_i = _merge9(body, snout)
        hips = [(0.3, 0.42, 0.34), (-0.3, 0.42, 0.34), (0.3, 0.42, -0.36), (-0.3, 0.42, -0.36)]
        legs = [_leg_part(0.09, 0.06, 0.42, T_BARK) for _ in range(4)]
        return body_v, body_i, hips, legs
    return _cached("boar", build)


def fire_model(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    def build():
        parts = []
        for a in (0.4, 1.9, 3.4):
            log, li = _tube(0.11, 0.09, 1.0, radial=6, rings=2, tile=T_BARK, cy=-0.5)
            rot_y = np.array([[np.cos(a), 0, np.sin(a), 0], [0, 1, 0, 0.0], [-np.sin(a), 0, np.cos(a), 0]])
            lay = _mul3(_move(0.0, 0.14, 0.0), _rot_z(np.pi / 2.0 - 0.2))
            parts.append((_xform9(log, _mul3(rot_y, lay)), li))
        for i in range(6):
            a = i * np.pi / 3.0
            stone = _sphere_part(0.14, 0.6, np.cos(a) * 0.75, 0.08, np.sin(a) * 0.75, T_ROCK, subdiv=0, amp=0.03, seed=40 + i)
            parts.append(stone)
        return _merge9(*parts)
    return _cached("fire", build)


def _rot_y_local(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0]])


# ------------------------------------------------------------------ ландшафт и пропсы
def build_terrain_chunk(world: "World", cx: int, cz: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(cx * CHUNK, cx * CHUNK + CHUNK + 1, dtype=np.float64)
    zs = np.arange(cz * CHUNK, cz * CHUNK + CHUNK + 1, dtype=np.float64)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    hh = world.h[cx * CHUNK:cx * CHUNK + CHUNK + 1, cz * CHUNK:cz * CHUNK + CHUNK + 1]
    nx = world.nx[cx * CHUNK:cx * CHUNK + CHUNK + 1, cz * CHUNK:cz * CHUNK + CHUNK + 1]
    ny = world.ny[cx * CHUNK:cx * CHUNK + CHUNK + 1, cz * CHUNK:cz * CHUNK + CHUNK + 1]
    nz = world.nz[cx * CHUNK:cx * CHUNK + CHUNK + 1, cz * CHUNK:cz * CHUNK + CHUNK + 1]
    splat = ground_splat(hh, ny)
    m = hh.size
    verts = np.empty((m, 11), dtype=np.float32)
    verts[:, 0] = xx.ravel()
    verts[:, 1] = hh.ravel()
    verts[:, 2] = zz.ravel()
    verts[:, 3] = nx.ravel()
    verts[:, 4] = ny.ravel()
    verts[:, 5] = nz.ravel()
    verts[:, 6:9] = splat.reshape(m, 3)
    verts[:, 9] = xx.ravel() * 0.25
    verts[:, 10] = zz.ravel() * 0.25
    lix, liz = np.meshgrid(np.arange(CHUNK), np.arange(CHUNK), indexing="ij")
    a = (lix * (CHUNK + 1) + liz).ravel()
    b, c, d = a + 1, a + (CHUNK + 1), a + (CHUNK + 1) + 1
    idx = np.stack([a, b, d, a, d, c], axis=1).ravel()
    return verts, idx.astype(np.uint32)


def gen_props(world: "World", seed: int) -> dict:
    rng = np.random.default_rng(seed)
    props: dict = {}
    for cx in range(WORLD_CHUNKS):
        for cz in range(WORLD_CHUNKS):
            plist = []
            for _ in range(42):
                x = float(rng.uniform(cx * CHUNK + 2, (cx + 1) * CHUNK - 2))
                z = float(rng.uniform(cz * CHUNK + 2, (cz + 1) * CHUNK - 2))
                h = world.height(x, z)
                if h < WATER_LEVEL + 0.4 or h > 26.0:
                    continue
                ix, iz = int(x), int(z)
                if world.ny[ix, iz] < 0.86:
                    continue
                f = float(world.forest[ix, iz])
                r = float(rng.uniform())
                sc = float(rng.uniform())
                if f > 0.52 and r < 0.55:
                    plist.append({"type": "tree", "x": x, "y": h, "z": z,
                                  "scale": 0.9 + 0.7 * sc, "yaw": float(rng.uniform(0, 6.28)),
                                  "variant": int((x * 7 + z * 3) % 3), "hp": 6,
                                  "alive": True, "r": 0.42})
                elif f > 0.30 and r < 0.70:
                    plist.append({"type": "bush", "x": x, "y": h, "z": z,
                                  "scale": 0.9 + 0.4 * sc, "yaw": float(rng.uniform(0, 6.28)),
                                  "variant": int((x * 5 + z * 7) % 3), "hp": 2,
                                  "alive": True, "berries": 2, "r": 0.0})
                elif r < 0.85:
                    plist.append({"type": "rock", "x": x, "y": h, "z": z,
                                  "scale": 0.75 + 0.65 * sc, "yaw": float(rng.uniform(0, 6.28)),
                                  "variant": int((x * 11 + z * 13) % 3), "hp": 4,
                                  "alive": True, "r": 0.5})
            props[(cx, cz)] = plist
    return props


TEMPLATES = {"tree": tree_model, "rock": rock_model, "bush": bush_model}


def build_prop_verts(plist: list) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for pr in plist:
        if not pr["alive"]:
            continue
        vi, fi = TEMPLATES[pr["type"]](pr["variant"])
        v = vi.copy()
        scale, yaw = pr["scale"], pr["yaw"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        v[:, 0:3] = (v[:, 0:3] * scale) @ ry.T + np.array([pr["x"], pr["y"] - 0.05, pr["z"]])
        parts.append((v, fi))
    if not parts:
        empty_v = np.zeros((0, 9), dtype=np.float32)
        return empty_v, np.zeros((0, 3), dtype=np.uint32)
    return _merge9(*parts)


# -------------------------------------------------------- пиксельный шрифт 5x7

GLYPH_W, GLYPH_H = 5, 7
CELL_W, CELL_H = 12, 16
_G = {}


def _g(key, rows):
    _G[key] = rows


def build_glyphs():
    _g("А", [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"])
    _g("Б", ["####.", "#....", "#....", "####.", "....#", "#...#", ".###."])
    _g("В", ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."])
    _g("Г", ["#####", "#....", "#....", "#....", "#....", "#....", "#...."])
    _g("Д", ["..###", "..#.#", "..#.#", "..#.#", "#####", "#...#", ".###."])
    _g("Е", ["#####", "#....", "#....", "####.", "#....", "#....", "####."])
    _g("Ё", ["#.#.#", "#####", "#....", "####.", "#....", "#....", "####."])
    _g("Ж", ["#.#.#", "#.#.#", "#.#.#", ".###.", "#.#.#", "#.#.#", "#.#.#"])
    _g("З", [".###.", "#...#", "#..#.", "..#..", "#..#.", "#...#", ".###."])
    _g("И", ["#...#", "#..##", "#..##", "#.#.#", "##..#", "#...#", "#...#"])
    _g("Й", [".#.#.", "#...#", "#..##", "#..##", "#.#.#", "##..#", "#...#"])
    _g("К", ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"])
    _g("Л", ["..###", "..#.#", "..#.#", "..#.#", "#..#.", "#..#.", "#..#."])
    _g("М", ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"])
    _g("Н", ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"])
    _g("О", [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."])
    _g("П", ["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#"])
    _g("Р", ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."])
    _g("С", [".####", "#....", "#....", "#....", "#....", "#....", ".####"])
    _g("Т", ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."])
    _g("У", ["#...#", "#...#", "#...#", ".####", "....#", "#...#", ".###."])
    _g("Ф", [".###.", "#.#.#", "#.#.#", ".###.", "..#..", "..#..", "..#.."])
    _g("Х", ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"])
    _g("Ц", ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "..###"])
    _g("Ч", ["#...#", "#...#", "#...#", ".####", "....#", "#...#", ".###."])
    _g("Ш", ["#####", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#####"])
    _g("Щ", ["#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "..###"])
    _g("Ъ", ["#...#", "#...#", "#.###", ".###.", "#...#", "#...#", ".###."])
    _g("Ы", ["#.###", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.###", "#...."])
    _g("Ь", ["#####", "#...#", "#...#", "#...#", "#...#", ".###.", "....."])
    _g("Э", [".####", "....#", "....#", ".###.", "....#", "....#", ".####"])
    _g("Ю", ["#.###", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.###"])
    _g("Я", [".####", ".#..#", ".#..#", ".####", "..#..", ".#...", "#...."])
    _g("0", [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."])
    _g("1", ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."])
    _g("2", [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"])
    _g("3", ["####.", "....#", "....#", ".###.", "....#", "....#", "####."])
    _g("4", ["..#..", ".##..", "#.#..", "#..#.", "#####", "..#..", "..#.."])
    _g("5", ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."])
    _g("6", ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."])
    _g("7", ["#####", "....#", "...#.", "..#..", "..#..", "..#..", "..#.."])
    _g("8", [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."])
    _g("9", [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."])
    _g(" ", ["....."] * 7)
    _g(":", [".....", "..#..", ".....", "..#..", ".....", "..#..", "....."])
    _g(".", [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."])
    _g(",", [".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."])
    _g("-", [".....", ".....", ".....", "#####", ".....", ".....", "....."])
    _g("!", ["..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."])
    _g("%", ["##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"])
    _g("/", ["....#", "...#.", "...#.", "..#..", "..#..", ".#...", "#...."])
    _g("(", ["..#..", ".#...", "#....", "#....", "#....", ".#...", "..#.."])
    _g(")", ["..#..", "...#.", "....#", "....#", "....#", "...#.", "..#.."])
    _g("+", [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."])
    _g("A", [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"])
    _g("C", [".####", "#....", "#....", "#....", "#....", "#....", ".####"])
    _g("D", ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."])
    _g("E", ["#####", "#....", "#....", "####.", "#....", "#....", "####."])
    _g("F", ["#####", "#....", "#....", "####.", "#....", "#....", "#...."])
    _g("H", ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"])
    _g("I", ["..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."])
    _g("K", ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"])
    _g("M", ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"])
    _g("P", ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."])
    _g("S", [".####", "#....", "#....", ".###.", "....#", "....#", "####."])
    _g("T", ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."])
    _g("W", ["#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", "#.#.#", ".#.#."])
    _g("X", ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"])
    _g("B", ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."])


def build_font_atlas() -> tuple[np.ndarray, dict]:
    build_glyphs()
    chars = sorted(_G.keys())
    cols = 16
    rows = (len(chars) + cols - 1) // cols
    atlas = np.zeros((rows * CELL_H, cols * CELL_W), dtype=np.uint8)
    index = {}
    for i, ch in enumerate(chars):
        gx = np.array([[1 if c == "#" else 0 for c in row] for row in _G[ch]], dtype=np.uint8)
        big = np.kron(gx, np.ones((2, 2), dtype=np.uint8))
        r, c = divmod(i, cols)
        y0, x0 = r * CELL_H + 1, c * CELL_W + 1
        atlas[y0:y0 + 14, x0:x0 + 10] = big * 255
        index[ch] = (x0, y0)
    return atlas, index


def text_width(s: str, scale: int = 2) -> int:
    return len(s) * CELL_W * scale


# ------------------------------------------------------------------ матрицы


def perspective(fov_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    fwd = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                    -math.cos(yaw) * math.cos(pitch)])
    f = fwd / np.linalg.norm(fwd)
    s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ortho_px(w: int, h: int) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = 2.0 / w
    m[1, 1] = -2.0 / h
    m[0, 3] = -1.0
    m[1, 3] = 1.0
    return m


def frustum_planes(pv: np.ndarray) -> np.ndarray:
    rows = np.stack([pv[3] + pv[0], pv[3] - pv[0], pv[3] + pv[1], pv[3] - pv[1], pv[2], pv[3]])
    n = np.linalg.norm(rows[:, :3], axis=1)
    return rows / n[:, None]


def sphere_in_frustum(planes: np.ndarray, c: np.ndarray, r: float) -> bool:
    return bool(np.all(planes @ np.append(c, 1.0) > -r))


def model_rot_y(yaw: float, x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def sun_state(t: float) -> dict:
    t01 = (t / DAY_LEN) % 1.0
    ang = 2.0 * math.pi * t01
    sy = math.sin(ang)
    day_k = float(np.clip(sy * 3.2, 0.0, 1.0))
    sunset_k = float(np.clip(1.0 - abs(sy) * 5.0, 0.0, 1.0)) * (1.0 if sy > -0.35 else 0.0)
    sd = np.array([math.cos(ang) * 0.9, max(sy, -0.2), 0.35])
    sd /= np.linalg.norm(sd)
    night_l = np.array([0.22, 0.26, 0.38])
    day_l = np.array([1.0, 0.97, 0.88])
    sunset_l = np.array([1.0, 0.6, 0.34])
    light = night_l * (1 - day_k) + (day_l * (1 - sunset_k) + sunset_l * sunset_k) * day_k
    night_s = np.array([0.016, 0.023, 0.05])
    day_s = np.array([0.45, 0.66, 0.92])
    sunset_s = np.array([0.85, 0.5, 0.30])
    sky = night_s * (1 - day_k) + (day_s * (1 - sunset_k) + sunset_s * sunset_k) * day_k
    amb = 0.14 + 0.30 * day_k
    return {"t01": t01, "day": bool(sy > -0.02), "sun_dir": sd, "light": light,
            "sky": sky, "amb": amb, "hour": (t01 * 24.0 + 6.0) % 24.0}


def sun_elev(st: dict) -> float:
    return math.sin(2.0 * math.pi * ((st["time"] / DAY_LEN) % 1.0))



def mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def model_translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def model_rot_x(a: float) -> np.ndarray:
    m = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


# ------------------------------------------------------------ игровая логика v2
HOTBAR = ("ТОПОР", "КОПЬЁ", "ЕДА", "РУКА")


def new_state(seed=None) -> dict:
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 10 ** 9))
    seed = int(seed)
    world = World(seed)
    chunks = {k: {"props": v} for k, v in gen_props(world, seed).items()}
    sx = sz = WORLD / 2
    best = 1e9
    for ix in range(WORLD // 2 - 48, WORLD // 2 + 48, 2):
        for iz in range(WORLD // 2 - 48, WORLD // 2 + 48, 2):
            i0, i1 = int(ix), int(iz)
            if (world.h[i0, i1] > WATER_LEVEL + 1.5 and world.h[i0, i1] < 12
                    and world.forest[i0, i1] > 0.5 and world.ny[i0, i1] > 0.9):
                d = abs(ix - WORLD / 2) + abs(iz - WORLD / 2)
                if d < best:
                    best = d
                    sx, sz = float(ix) + 0.5, float(iz) + 0.5
    rng = np.random.default_rng(seed + 555)
    boars = []
    tries = 0
    while len(boars) < 4 and tries < 200:
        tries += 1
        bx = float(rng.uniform(8, WORLD - 8))
        bz = float(rng.uniform(8, WORLD - 8))
        bh = world.height(bx, bz)
        if bh < WATER_LEVEL + 1.0 or bh > 20.0:
            continue
        boars.append({"x": bx, "z": bz, "yaw": float(rng.uniform(0, 6.28)),
                      "hp": 6, "wt": 0.0, "goal": 0.0, "flee": 0.0, "y": bh})
    return {
        "seed": seed, "world": world, "chunks": chunks,
        "px": sx, "py": world.height(sx, sz) + EYE, "pz": sz,
        "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.7, "pitch": -0.08,
        "on_ground": True, "swim": False,
        "hp": 100.0, "hunger": 85.0, "thirst": 80.0,
        "inv": {"дерево": 0, "камень": 0, "ягоды": 0, "сырое мясо": 0, "жареное мясо": 0},
        "tools": {"топор": False, "копьё": False},
        "selected": 3,
        "fires": [], "wolves": [], "boars": boars, "pickups": [],
        "time": DAY_LEN * 0.25, "days": 1, "attack_cd": 0.0,
        "messages": [], "craft_open": False, "inv_open": False,
        "dead": False, "paused": False,
        "spawn_pending": 3.0, "started": False, "show_help": False,
        "cold": False, "was_night": False,
    }


def message(st: dict, text: str):
    st["messages"] = [(text, 3.0)] + st["messages"][:3]


def in_water_body(st: dict) -> bool:
    return st["py"] - EYE < WATER_LEVEL - 0.3


def is_swimming(st: dict) -> bool:
    g = st["world"].height(st["px"], st["pz"])
    return g < WATER_LEVEL - 1.0 and st["py"] - EYE < WATER_LEVEL - 0.3


def collides_with_props(st: dict, nx: float, nz: float, py: float) -> tuple[float, float]:
    ox, oz = nx, nz
    cx, cz = int(nx // CHUNK), int(nz // CHUNK)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            ch = st["chunks"].get((cx + dx, cz + dz))
            if not ch:
                continue
            for pr in ch["props"]:
                if not pr["alive"] or pr["r"] < 0.3:
                    continue
                if pr["y"] > py + 2.5 or pr["y"] < py - 3.5:
                    continue
                ddx, ddz = ox - pr["x"], oz - pr["z"]
                d2 = ddx * ddx + ddz * ddz
                rr = pr["r"] * pr["scale"] + PLAYER_R
                if 1e-6 < d2 < rr * rr:
                    d = math.sqrt(d2)
                    ox += ddx / d * (rr - d)
                    oz += ddz / d * (rr - d)
    return float(np.clip(ox, 2.0, WORLD - 2.0)), float(np.clip(oz, 2.0, WORLD - 2.0))


def update_player(st: dict, dt: float, move: tuple[float, float], sprint: bool, jump: bool):
    if st["dead"]:
        return
    swim = is_swimming(st)
    st["swim"] = swim
    yaw = st["yaw"]
    fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    wish = fwd * move[0] + right * move[1]
    wl = float(np.linalg.norm(wish))
    if wl > 1e-6:
        wish /= wl
    speed = SWIM_SPEED if swim else (SPRINT if (sprint and move[0] > 0) else WALK)
    k = min(1.0, (10.0 if swim else 14.0) * dt)
    st["vx"] += (wish[0] * speed - st["vx"]) * k
    st["vz"] += (wish[2] * speed - st["vz"]) * k
    if swim:
        surface = WATER_LEVEL - 0.3
        st["vy"] += ((surface - (st["py"] - EYE + 0.4)) * 8.0 - st["vy"] * 2.5) * dt
        if jump:
            st["vy"] = max(st["vy"], 1.6)
        if sprint:
            st["vy"] -= 2.0 * dt
        if (st["py"] - EYE) < WATER_LEVEL - 2.6 and move[0] < -0.1:
            st["vy"] -= 3.0 * dt
    else:
        st["vy"] -= GRAV * dt
        if jump and st["on_ground"]:
            st["vy"] = JUMP_V
            st["on_ground"] = False
    nx = st["px"] + st["vx"] * dt
    nz = st["pz"] + st["vz"] * dt
    ny = st["py"] + st["vy"] * dt
    nx, nz = collides_with_props(st, nx, nz, st["py"])
    g = st["world"].height(nx, nz)
    st["px"], st["pz"], st["py"] = nx, nz, ny
    if st["py"] <= g + EYE:
        st["py"], st["vy"], st["on_ground"] = g + EYE, 0.0, True
    else:
        st["on_ground"] = False
    if wl > 1e-6 and not swim:
        drain = 1.0 + (0.9 if sprint else 0.0)
        st["hunger"] -= dt * 0.30 * drain
        st["thirst"] -= dt * 0.25 * drain


def update_survival(st: dict, dt: float):
    if st["dead"]:
        return
    st["hunger"] = max(0.0, st["hunger"] - dt * 0.16)
    st["thirst"] = max(0.0, st["thirst"] - dt * 0.22)
    dmg = 0.0
    if st["hunger"] <= 0:
        dmg += 1.4
    if st["thirst"] <= 0:
        dmg += 1.9
    near_fire = any((st["px"] - f["x"]) ** 2 + (st["pz"] - f["z"]) ** 2 < 36.0
                    for f in st["fires"])
    night = sun_elev(st) < -0.12
    st["cold"] = bool(night and not near_fire)
    if st["cold"]:
        dmg += 1.1
    if dmg > 0:
        st["hp"] = max(0.0, st["hp"] - dt * dmg)
        if st["hp"] <= 0:
            st["dead"] = True
            st["craft_open"] = False
            st["inv_open"] = False
    elif st["hunger"] > 30 and st["thirst"] > 30:
        st["hp"] = min(100.0, st["hp"] + dt * 0.7)


def update_fires(st: dict, dt: float):
    for f in st["fires"]:
        f["fuel"] -= dt
    before = len(st["fires"])
    st["fires"] = [f for f in st["fires"] if f["fuel"] > 0]
    if len(st["fires"]) < before:
        message(st, "КОСТЁР ПОГАС. НУЖЕН НОВЫЙ (C)")


def update_wolves(st: dict, dt: float):
    if st["dead"]:
        return
    elev = sun_elev(st)
    target = 0 if elev > 0.05 else min(2 + st["days"] // 2, 6)
    if len(st["wolves"]) < target:
        st["spawn_pending"] -= dt
        if st["spawn_pending"] <= 0:
            st["spawn_pending"] = 6.0
            rngw = np.random.default_rng()
            ang = st["yaw"] + math.pi + float(rngw.uniform(-1.2, 1.2))
            d = float(rngw.uniform(28, 44))
            wx = float(np.clip(st["px"] + math.sin(ang) * d, 3, WORLD - 3))
            wz = float(np.clip(st["pz"] + math.cos(ang) * d, 3, WORLD - 3))
            st["wolves"].append({"x": wx, "z": wz, "yaw": ang, "hp": 3, "goal": ang,
                                 "wt": 0.0, "cd": 0.0, "hit_t": 0.0, "run": 0.0})
    alive = []
    for w in st["wolves"]:
        dx, dz = st["px"] - w["x"], st["pz"] - w["z"]
        dist = math.hypot(dx, dz) + 1e-6
        if elev > 0.25 and w["hp"] >= 3 and dist > 26:
            continue
        if w["hit_t"] > 0:
            w["hit_t"] -= dt
        night = elev < 0.05
        if dist < 26:
            sp = 5.4 if dist > 3 else 2.0
            w["x"] += dx / dist * sp * dt
            w["z"] += dz / dist * sp * dt
            w["yaw"] = math.atan2(dx, dz)
            w["run"] = 1.0
        elif night or w["hp"] < 3:
            sp = 1.6
            w["x"] += dx / dist * sp * dt
            w["z"] += dz / dist * sp * dt
            w["yaw"] = math.atan2(dx, dz)
            w["run"] = 0.5
        else:
            w["wt"] -= dt
            if w["wt"] <= 0:
                w["wt"] = float(np.random.default_rng().uniform(2.0, 5.0))
                w["goal"] = float(np.random.default_rng().uniform(0, 2 * math.pi))
            w["x"] += math.cos(w["goal"]) * 1.6 * dt
            w["z"] += math.sin(w["goal"]) * 1.6 * dt
            w["yaw"] = w["goal"]
            w["run"] = 0.4
        if st["world"].height(w["x"], w["z"]) < WATER_LEVEL - 0.5:
            w["x"] -= dx / dist * 3.2 * dt
        w["y"] = st["world"].height(w["x"], w["z"])
        w["cd"] -= dt
        if dist < 1.7 and w["cd"] <= 0:
            w["cd"] = 1.3
            st["hp"] = max(0.0, st["hp"] - 9.0)
            message(st, "ВОЛК АТАКУЕТ! (-9 ЗДОРОВЬЯ)")
            if st["hp"] <= 0:
                st["dead"] = True
                st["craft_open"] = False
                st["inv_open"] = False
        if w["hp"] <= 0:
            for _ in range(2):
                st["pickups"].append({"x": w["x"], "y": w["y"], "z": w["z"], "item": "сырое мясо"})
            message(st, "ВОЛК ПОВЕРЖЕН! +2 СЫРОГО МЯСА")
            continue
        alive.append(w)
    st["wolves"] = alive


def update_boars(st: dict, dt: float):
    if st["dead"]:
        return
    alive = []
    for b in st["boars"]:
        dx, dz = st["px"] - b["x"], st["pz"] - b["z"]
        dist = math.hypot(dx, dz) + 1e-6
        if b["flee"] > 0 or dist < 4.0:
            b["flee"] = max(b["flee"], 2.5)
            b["flee"] -= dt
            sp = 6.0
            b["x"] -= dx / dist * sp * dt
            b["z"] -= dz / dist * sp * dt
            b["yaw"] = math.atan2(-dx, -dz)
        else:
            b["wt"] -= dt
            if b["wt"] <= 0:
                b["wt"] = float(np.random.default_rng().uniform(2.0, 6.0))
                b["goal"] = float(np.random.default_rng().uniform(0, 2 * math.pi))
            b["x"] += math.cos(b["goal"]) * 1.2 * dt
            b["z"] += math.sin(b["goal"]) * 1.2 * dt
            b["yaw"] = b["goal"]
        b["x"] = float(np.clip(b["x"], 2, WORLD - 2))
        b["z"] = float(np.clip(b["z"], 2, WORLD - 2))
        g = st["world"].height(b["x"], b["z"])
        if g < WATER_LEVEL - 0.5:
            b["x"] += math.cos(b["yaw"]) * 3.0 * dt
            b["z"] += math.sin(b["yaw"]) * 3.0 * dt
            g = st["world"].height(b["x"], b["z"])
        b["y"] = g
        if b["hp"] <= 0:
            for _ in range(3):
                st["pickups"].append({"x": b["x"], "y": b["y"], "z": b["z"], "item": "сырое мясо"})
            message(st, "КАБАН УБИТ! +3 СЫРОГО МЯСА")
            continue
        alive.append(b)
    st["boars"] = alive


def interact_target(st: dict):
    px, py, pz = st["px"], st["py"], st["pz"]
    yaw, pitch = st["yaw"], st["pitch"]
    fwd = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                    -math.cos(yaw) * math.cos(pitch)])
    best, best_d = None, 1e18
    for pk in st["pickups"]:
        d = math.hypot(pk["x"] - px, pk["z"] - pz)
        if d < 2.2 and d < best_d:
            best, best_d = ("pickup", pk), d
    if best is None:
        cx, cz = int(px // CHUNK), int(pz // CHUNK)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                ch = st["chunks"].get((cx + dx, cz + dz))
                if not ch:
                    continue
                for pr in ch["props"]:
                    if not pr["alive"] or pr["type"] not in ("bush",):
                        continue
                    if pr.get("berries", 0) == 0:
                        continue
                    dvec = np.array([pr["x"] - px, pr["y"] + 0.5 - py, pr["z"] - pz])
                    d = float(np.linalg.norm(dvec))
                    if d > REACH:
                        continue
                    if float(dvec @ fwd) / max(d, 1e-6) < 0.35:
                        continue
                    if d < best_d:
                        best, best_d = ("prop", pr), d
    if best is None:
        ahead_h = st["world"].height(px - math.sin(yaw) * 1.3, pz - math.cos(yaw) * 1.3)
        if ahead_h < WATER_LEVEL + 0.25:
            best = ("water", None)
    return best


def hit_target(st: dict, reach: float):
    """Ближайшая цель (животное или дерево/камень) в конусе взгляда."""
    px, _, pz = st["px"], st["py"], st["pz"]
    yaw = st["yaw"]
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    best, best_d = None, 1e18
    for lst, kindtag in ((st["wolves"], "wolf"), (st["boars"], "boar")):
        for an in lst:
            dx, dz = an["x"] - px, an["z"] - pz
            d = math.hypot(dx, dz) + 1e-6
            if d > reach:
                continue
            if (dx / d) * fx + (dz / d) * fz < 0.4:
                continue
            if d < best_d:
                best_d = d
                best = ("animal", kindtag, an)
    if best is None:
        cx, cz = int(px // CHUNK), int(pz // CHUNK)
        for dx2 in (-1, 0, 1):
            for dz2 in (-1, 0, 1):
                ch = st["chunks"].get((cx + dx2, cz + dz2))
                if not ch:
                    continue
                for pr in ch["props"]:
                    if not pr["alive"] or pr["type"] not in ("tree", "rock"):
                        continue
                    hx, hz = pr["x"] - px, pr["z"] - pz
                    d = math.hypot(hx, hz) + 1e-6
                    if d - pr["r"] * pr["scale"] > reach:
                        continue
                    if (hx / d) * fx + (hz / d) * fz < 0.2:
                        continue
                    if d - pr["r"] < best_d:
                        best_d = d - pr["r"]
                        best = ("prop", pr["type"], pr)
    return best


def selected_weapon(st: dict) -> tuple[str, float, float]:
    slot = st.get("selected", 3)
    if slot == 0 and st["tools"]["топор"]:
        return "топор", 3.0, 2.7
    if slot == 1 and st["tools"]["копьё"]:
        return "копьё", 2.0, 3.0
    return "рука", 2.0, 2.2


def do_attack(st: dict):
    if st["dead"] or st["attack_cd"] > 0:
        return None
    slot = st.get("selected", 3)
    if slot == 2:
        eat_food(st)
        return None
    wname, wdmg, wreach = selected_weapon(st)
    st["attack_cd"] = 0.5 if wname == "топор" else (0.6 if wname == "копьё" else 0.45)
    tgt = hit_target(st, wreach)
    if tgt is None:
        return None
    kind, tag, obj = tgt
    inv = st["inv"]
    if kind == "animal":
        obj["hp"] -= wdmg
        if "hit_t" in obj:
            obj["hit_t"] = 0.25
        if tag == "boar":
            obj["flee"] = 4.0
        dx = obj["x"] - st["px"]
        dz = obj["z"] - st["pz"]
        d = math.hypot(dx, dz) + 1e-6
        obj["x"] += dx / d * 1.2
        obj["z"] += dz / d * 1.2
        return None
    obj["hp"] -= wdmg
    if tag == "tree":
        n = 3 if wname == "топор" else 1
        inv["дерево"] += n
        message(st, f"+{n} ДЕРЕВО")
    else:
        inv["камень"] += 2
        message(st, "+2 КАМНЯ")
    if obj["hp"] <= 0:
        obj["alive"] = False
        if tag == "tree":
            inv["дерево"] += 2
            message(st, "ДЕРЕВО ПОВЕРЖЕНО! +2 ДЕРЕВА")
        else:
            inv["камень"] += 1
        return (int(obj["x"] // CHUNK), int(obj["z"] // CHUNK))
    return (int(obj["x"] // CHUNK), int(obj["z"] // CHUNK))


def do_interact(st: dict):
    if st["dead"]:
        return None
    tgt = interact_target(st)
    if tgt is None:
        return None
    kind, obj = tgt
    inv = st["inv"]
    if kind == "pickup":
        inv[obj["item"]] += 1
        st["pickups"].remove(obj)
        message(st, "ВЗЯТО: " + obj["item"].upper())
        return None
    if kind == "water":
        st["thirst"] = min(100.0, st["thirst"] + 35.0)
        message(st, "ВЫ ПЬЁТЕ ВОДУ. ЖАЖДА УТЛЕНА")
        return None
    n = obj["berries"]
    inv["ягоды"] += n
    obj["berries"] = 0
    obj["hp"] -= 1
    message(st, f"+{n} ЯГОД")
    if obj["hp"] <= 0:
        obj["alive"] = False
    return (int(obj["x"] // CHUNK), int(obj["z"] // CHUNK))


def eat_food(st: dict):
    if st["dead"]:
        return
    inv = st["inv"]
    if inv["жареное мясо"] > 0:
        inv["жареное мясо"] -= 1
        st["hunger"] = min(100.0, st["hunger"] + 45.0)
        message(st, "СЪЕДЕНО ЖАРЕНОЕ МЯСО +45")
    elif inv["ягоды"] > 0:
        inv["ягоды"] -= 1
        st["hunger"] = min(100.0, st["hunger"] + 12.0)
        st["thirst"] = min(100.0, st["thirst"] + 8.0)
        message(st, "СЪЕДЕНЫ ЯГОДЫ +12")
    elif inv["сырое мясо"] > 0:
        inv["сырое мясо"] -= 1
        st["hunger"] = min(100.0, st["hunger"] + 8.0)
        message(st, "СЫРОЕ МЯСО +8. ЖАРИТЬ НА КОСТРЕ ЛУЧШЕ!")
    else:
        message(st, "НЕТ ЕДЫ")


def can_craft(st: dict, idx: int) -> bool:
    if not (0 <= idx < len(RECIPES)):
        return False
    return all(st["inv"][k] >= v for k, v in RECIPES[idx][1])


def try_craft(st: dict, idx: int) -> bool:
    if st["dead"] or not (0 <= idx < len(RECIPES)):
        return False
    name, cost = RECIPES[idx]
    if not can_craft(st, idx):
        missing = ", ".join(f"{k} {v - st['inv'][k]}" for k, v in cost if st["inv"][k] < v)
        message(st, "НЕ ХВАТАЕТ: " + missing)
        return False
    for k, v in cost:
        st["inv"][k] -= v
    if name == "ТОПОР":
        st["tools"]["топор"] = True
        st["selected"] = 0
        message(st, "СОЗДАН ТОПОР! СЛОТ 1, РУБИ ЛКМ")
    elif name == "КОПЬЁ":
        st["tools"]["копьё"] = True
        message(st, "СОЗДАНО КОПЬЁ! СЛОТ 2")
    else:
        yaw = st["yaw"]
        fx = float(np.clip(st["px"] - math.sin(yaw) * 2.0, 2, WORLD - 2))
        fz = float(np.clip(st["pz"] - math.cos(yaw) * 2.0, 2, WORLD - 2))
        st["fires"].append({"x": fx, "z": fz, "y": st["world"].height(fx, fz), "fuel": 130.0})
        message(st, "КОСТЁР ГОРИТ! ТЕПЛО У ОГНЯ. E У ОГНЯ - ЖАРИТЬ МЯСО")
    return True


def cook_at_fire(st: dict) -> bool:
    for f in st["fires"]:
        if (st["px"] - f["x"]) ** 2 + (st["pz"] - f["z"]) ** 2 < 6.25:
            if st["inv"]["сырое мясо"] > 0:
                st["inv"]["сырое мясо"] -= 1
                st["inv"]["жареное мясо"] += 1
                message(st, "МЯСО ПРИГОТОВЛЕНО! ЖМИ F ЧТОБЫ СЪЕСТЬ")
            else:
                message(st, "НЕТ СЫРОГО МЯСА. ОХОТЬСЯ ЛКМ")
            return True
    return False


def update_time(st: dict, dt: float):
    before_night = sun_elev(st) < -0.02
    st["time"] += dt
    after_night = sun_elev(st) < -0.02
    if st["time"] >= st["days"] * DAY_LEN:
        st["days"] = 1 + int(st["time"] // DAY_LEN)
    if after_night and not before_night:
        message(st, "НОЧЬ. ВОЛКИ ГОЛОДНЫ. ЖГИ КОСТЁР!")


def selftest() -> int:
    print("[1/7] пиксельный шрифт и атлас")
    atlas, index = build_font_atlas()
    assert atlas.dtype == np.uint8 and atlas.max() == 255
    missing = {ch for grp in UI_STRINGS for ch in grp if ch not in index}
    assert not missing, f"нет глифов: {missing}"
    print(f"      атлас {atlas.shape}, глифов {len(index)}")

    print("[2/7] текстурный атлас мира")
    ta = build_texture_atlas()
    assert ta.shape == (ATLAS_ROWS * TILE, ATLAS_COLS * TILE, 4) and ta.dtype == np.uint8
    assert ta[..., 3].max() == 255 and ta[..., 3].min() == 0
    for t in range(9):
        u0, v0 = atlas_tile_uv(t)
        assert 0 <= u0 < 1 and 0 <= v0 < 1
    print("      9 тайлов 64px, alpha-канал: OK")

    print("[3/7] шум и рельеф")
    w = World(1337)
    h = w.h
    assert np.all(np.isfinite(h)) and 0 <= h.min() and h.max() <= 38
    assert h.std() > 1.0, "рельеф плоский?"
    water_p = float((h < WATER_LEVEL).mean())
    assert water_p > 0.02, "нужны озёра"
    w2 = World(9999)
    assert float(np.abs(w2.h - w.h).mean()) > 0.5, "миры с разными сидами одинаковы?"
    sp = ground_splat(h, w.ny)
    assert sp.shape == h.shape + (3,) and np.allclose(sp.sum(axis=-1), 1.0)
    print(f"      высоты {h.min():.0f}..{h.max():.0f}, вода {water_p * 100:.0f}%, сиды различаются")

    print("[4/7] модели и меши")
    tv, ti = tree_model(1)
    assert tv.shape[1] == 9 and ti.dtype == np.uint32 and ti.max() < len(tv)
    for name, fn in (("rock", rock_model), ("bush1", lambda: bush_model(1)),
                     ("wolf", lambda: wolf_model()[:2]), ("boar", lambda: boar_model()[:2]),
                     ("fire", fire_model)):
        v, i = fn()
        assert v.shape[1] == 9 and np.isfinite(v).all() and i.max() < len(v), name
    _, _, hips, legs = wolf_model()
    assert len(hips) == 4 and len(legs) == 4 and legs[0][0][:, 1].max() <= 0.01
    va, ia = build_terrain_chunk(w, 3, 3)
    assert va.dtype == np.float32 and va.shape[1] == 11 and ia.dtype == np.uint32
    assert ia.shape[0] == CHUNK * CHUNK * 6 and np.all(np.isfinite(va))
    props = gen_props(w, 1337)
    total = sum(len(p) for p in props.values())
    pv, pi = build_prop_verts(props[(3, 3)])
    assert pi.max() < len(pv) and np.isfinite(pv).all()
    print(f"      дерево {ti.shape[0] // 3} трис, волк + лапы OK, карта {total} пропсов")

    print("[5/7] матрицы и фрустум")
    aspect = W_SIZE[0] / W_SIZE[1]
    P = perspective(FOV, aspect, 0.1, 260.0)
    eye = np.array([128.0, 12.0, 128.0])
    PVm = P @ look_at(eye, 0.0, 0.0)
    ndc = PVm @ np.append(eye + np.array([0, 0, -10.0]), 1.0)
    ndc /= ndc[3]
    assert abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4 and -1 < ndc[2] < 1
    planes = frustum_planes(PVm)
    assert sphere_in_frustum(planes, eye + np.array([0, 0, -30.0]), 5.0)
    assert not sphere_in_frustum(planes, eye + np.array([0, 0, 30.0]), 5.0)
    lm = mat4_mul(model_rot_y(0.7, 5, 6, 7), mat4_mul(model_translate(1, 2, 3), model_rot_x(0.3)))
    t = lm @ np.array([0, 0, 0, 1])
    c7, s7 = math.cos(0.7), math.sin(0.7)
    assert np.allclose(t[:3], [5 + c7 * 1 + s7 * 3, 8, 7 - s7 * 1 + c7 * 3])
    print("      NDC, отсечение, u_model-композиция: OK")

    print("[6/7] игрок: движение, плавание, добыча, крафт")
    st = new_state(1337)
    g0 = st["world"].height(st["px"], st["pz"])
    assert abs(st["py"] - (g0 + EYE)) < 1e-6 and st["selected"] == 3
    update_player(st, 0.016, (1.0, 0.0), False, False)
    update_player(st, 0.016, (1.0, 0.0), False, False)
    assert st["hunger"] < 85.0
    for _ in range(400):
        update_player(st, 0.016, (0.0, 0.0), False, False)
    g = st["world"].height(st["px"], st["pz"])
    assert abs(st["py"] - (g + EYE)) < 1e-6
    wp = min((((int(x), int(z))) for x in range(WORLD // 2, WORLD, 2)
              for z in range(WORLD // 2, WORLD, 2)
              if w.h[int(x), int(z)] < WATER_LEVEL - 2.0), default=None)
    if wp is not None:
        st["px"], st["pz"] = wp[0] + 0.5, wp[1] + 0.5
        st["py"] = w.height(st["px"], st["pz"]) + EYE
        st["vy"] = -2.0
        for _ in range(90):
            update_player(st, 0.016, (1.0, 0.0), False, False)
        assert st["swim"], "должен плыть в глубоководье"
        assert (st["py"] - EYE) > WATER_LEVEL - 2.0, "плавучесть держит у поверхности"
    st["inv"]["дерево"], st["inv"]["камень"] = 3, 2
    assert try_craft(st, 0) and st["tools"]["топор"] and st["selected"] == 0
    assert st["inv"]["дерево"] == 0
    st["inv"]["дерево"], st["inv"]["камень"] = 5, 4
    assert try_craft(st, 2) and len(st["fires"]) == 1
    st["inv"]["сырое мясо"] = 2
    f0 = st["fires"][0]
    st["px"], st["pz"] = f0["x"] + 0.8, f0["z"]
    assert cook_at_fire(st) and st["inv"]["жареное мясо"] == 1
    eat_food(st)
    assert st["hunger"] > 85.0
    tree = next(pr for ch in st["chunks"].values() for pr in ch["props"]
                if pr["alive"] and pr["type"] == "tree")
    st["px"], st["pz"] = tree["x"] + 2.0, tree["z"]
    st["py"] = st["world"].height(st["px"], st["pz"]) + EYE
    st["yaw"] = math.atan2(-(tree["x"] - st["px"]), -(tree["z"] - st["pz"]))
    st["selected"] = 0
    wood0 = st["inv"]["дерево"]
    st["attack_cd"] = 0.0
    do_attack(st); st["attack_cd"] = 0.0
    do_attack(st)
    assert tree["hp"] == 0 and not tree["alive"], "топор валит дерево за 2 удара"
    assert st["inv"]["дерево"] == wood0 + 8, "рубка даёт древесину"
    print("      ходьба, плавание, топор=2 удара по дереву, крафт, еда: OK")

    print("[7/7] животные: волки ночью, кабаны днём")
    st2 = new_state(1337)
    st2["time"] = DAY_LEN * 0.75
    assert sun_elev(st2) < -0.02
    st2["spawn_pending"] = 0.0
    update_wolves(st2, 0.1)
    assert len(st2["wolves"]) == 1
    st3 = new_state(1337)
    assert len(st3["boars"]) >= 1
    b0 = st3["boars"][0]
    st3["px"], st3["pz"] = b0["x"] + 1.0, b0["z"]
    b0["hp"] = 1
    st3["selected"] = 0
    st3["tools"]["топор"] = True
    st3["yaw"] = math.atan2(-(b0["x"] - st3["px"]), -(b0["z"] - st3["pz"]))
    st3["attack_cd"] = 0.0
    do_attack(st3)
    update_boars(st3, 0.01)
    assert b0["hp"] <= 0 and len(st3["pickups"]) == 3
    b1 = next(b for b in st3["boars"] if b["hp"] > 0)
    bx0 = b1["x"]
    st3["px"], st3["pz"] = b1["x"] + 0.7, b1["z"]
    for _ in range(3):
        update_boars(st3, 0.2)
    assert math.hypot(b1["x"] - bx0, 0) > 0.1, "кабан убегает"
    print("      волк ночью спавнится, кабан убит -> +3 мяса, паника: OK")
    print()
    print("SELFTEST OK v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
