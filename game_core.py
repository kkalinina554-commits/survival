"""Ядро игры: шум, рельеф, пропсы, шрифт, матрицы, игровая логика. Без OpenGL."""
import math

import numpy as np

W_SIZE = (1280, 720)
FOV = 75.0
WORLD_SEED = 1337
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
REACH = 2.7
PLAYER_R = 0.38

RECIPES = [
    ("ТОПОР", [("дерево", 3), ("камень", 2)]),
    ("КОПЬЁ", [("дерево", 2), ("камень", 1)]),
    ("КОСТЁР", [("дерево", 5), ("камень", 4)]),
]

UI_STRINGS = (
    "ЗДОРОВЬЕ ГОЛОД ЖАЖДА ТЕПЛО ДЕРЕВО КАМЕНЬ ЯГОДЫ СЫРОЕ МЯСО ЖАРЕНОЕ МЯСО",
    "ДЕНЬ НОЧЬ ПАУЗА КРАФТ:",
    "1 - ТОПОР: 3 ДЕРЕВА + 2 КАМНЯ",
    "2 - КОПЬЁ: 2 ДЕРЕВА + 1 КАМЕНЬ",
    "3 - КОСТЁР: 5 ДЕРЕВА + 4 КАМНЯ",
    "УЖЕ ЕСТЬ! НЕ ХВАТАЕТ:",
    "КОСТЁР ГОРИТ! ТЕПЛО У ОГНЯ. E У ОГНЯ - ПРИГОТОВИТЬ МЯСО",
    "+1 ДЕРЕВО +3 ДЕРЕВА +1 КАМЕНЬ +2 КАМНЯ +2 ЯГОДЫ +4 ЯГОДЫ",
    "ВЫ ПЬЁТЕ ВОДУ. ЖАЖДА УТЛЕНА",
    "СЪЕДЕНО ЖАРЕНОЕ МЯСО +45 СЪЕДЕНЫ ЯГОДЫ +12 СЫРОЕ МЯСО +8 НЕТ ЕДЫ",
    "МЯСО ПРИГОТОВЛЕНО! ЖМИ F ЧТОБЫ СЪЕСТЬ НЕТ СЫРОГО МЯСА. ОХОТЬСЯ ЛКМ",
    "ВЗЯТО СЫРОЕ МЯСО ВОЛК ПОВЕРЖЕН! +2 СЫРОГО МЯСА",
    "ГОЛОД! ЖАЖДА! ХОЛОД! РАЗВЕДИ КОСТЁР (C)",
    "НОЧЬ. ВОЛКИ ГОЛОДНЫ. ЖГИ КОСТЁР! КОСТЁР ПОГАС. НУЖЕН НОВЫЙ (C)",
    "ВЫ ПОГИБЛИ ВЫЖИЛ ДНЕЙ: ПРОБЕЛ - ЗАНОВО",
    "ЛЕС. ВЫЖИВАНИЕ WASD - ДВИЖЕНИЕ МЫШЬ - ОБЗОР SHIFT - БЕГ SPACE - ПРЫЖОК",
    "E - ДОБЫТЬ / ВЗЯТЬ / ПИТЬ ВОДУ ЛКМ - АТАКА (КОПЬЁ БЬЁТ СИЛЬНЕЕ)",
    "F - ЕСТЬ C - КРАФТ H - ПОМОЩЬ ESC - ПАУЗА",
    "ЗАДАЧА: ВЕЧЕРОМ РАЗВЕДИ КОСТЁР. НОЧЬЮ - ВОЛКИ. ПЕЙ У ВОДЫ, ЕШЬ ЯГОДЫ.",
    "ПРОБЕЛ - НАЧАТЬ ДЕНЬ ПЕРВЫЙ",
)


HELP_LINES = (
    "WASD - ДВИЖЕНИЕ, МЫШЬ - ОБЗОР, SHIFT - БЕГ, SPACE - ПРЫЖОК",
    "E - ДОБЫТЬ / ВЗЯТЬ / ПИТЬ ВОДУ У БЕРЕГА, ЛКМ - АТАКА",
    "F - ЕСТЬ (ЯГОДЫ И МЯСО), C - КРАФТ, H - ПОМОЩЬ, ESC - ПАУЗА",
    "ЗАДАЧА: НАРУБИ ДЕРЕВЬЕВ И КАМНЯ, ВЕЧЕРОМ - КОСТЁР.",
    "НОЧЬЮ ПРИХОДЯТ ВОЛКИ. У ОГНЯ ТЕПЛО, БЕЗ НЕГО ХОЛОД.",
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
    def __init__(self, seed: int = WORLD_SEED):
        self.seed = seed
        n = WORLD + 1
        gx, gz = np.meshgrid(np.arange(n, dtype=np.float64),
                             np.arange(n, dtype=np.float64), indexing="ij")
        n1 = fbm(gx, gz, 1.0 / 57.0, 4, seed)
        n2 = fbm(gx, gz, 1.0 / 19.0, 3, seed + 7)
        self.h = np.clip((n1 - 0.36) * 46.0 + (n2 - 0.5) * 7.0, 0.0, 34.0)
        self.forest = fbm(gx, gz, 1.0 / 31.0, 3, seed + 13)
        self.tint = fbm(gx, gz, 1.0 / 9.0, 2, seed + 29)
        dhdx, dhdz = np.gradient(self.h)
        nl = np.sqrt(1.0 + dhdx * dhdx + dhdz * dhdz)
        self.nx, self.ny, self.nz = -dhdx / nl, 1.0 / nl, -dhdz / nl

    def height(self, x: float, z: float) -> float:
        x = float(np.clip(x, 0.0, WORLD - 1e-3))
        z = float(np.clip(z, 0.0, WORLD - 1e-3))
        ix, iz = int(x), int(z)
        tx, tz = x - ix, z - iz
        h00, h10 = self.h[ix, iz], self.h[ix + 1, iz]
        h01, h11 = self.h[ix, iz + 1], self.h[ix + 1, iz + 1]
        return float((h00 * (1 - tx) + h10 * tx) * (1 - tz)
                     + (h01 * (1 - tx) + h11 * tx) * tz)


def ground_color(h: np.ndarray, ny: np.ndarray, tint: np.ndarray) -> np.ndarray:
    sand = np.array([0.72, 0.66, 0.47])
    grass = np.array([0.27, 0.50, 0.22]) * (0.85 + 0.3 * tint)[..., None]
    dirt = np.array([0.40, 0.30, 0.20])
    rock = np.array([0.48, 0.47, 0.47])
    col = np.where((h < WATER_LEVEL + 1.6)[..., None], sand, grass)
    slope_t = np.clip((1.0 - ny - 0.55) / 0.30, 0.0, 1.0)[..., None]
    col = col * (1 - slope_t) + rock * slope_t
    dirt_t = np.clip((1.0 - ny - 0.30) / 0.35, 0.0, 1.0)[..., None] * (1 - slope_t)
    col = col * (1 - dirt_t) + dirt * dirt_t
    return col


def build_terrain_vertices(world: World, cx: int, cz: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(cx * CHUNK, (cx + 1) * CHUNK + 1, dtype=np.int64)
    zs = np.arange(cz * CHUNK, (cz + 1) * CHUNK + 1, dtype=np.int64)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    H = world.h[X, Z]
    NX, NY, NZ = world.nx[X, Z], world.ny[X, Z], world.nz[X, Z]
    col = ground_color(H, NY, world.tint[X, Z]).astype(np.float32)
    n = H.size
    verts = np.empty((n, 9), dtype=np.float32)
    verts[:, 0] = X.ravel() + 0.5
    verts[:, 1] = H.ravel()
    verts[:, 2] = Z.ravel() + 0.5
    verts[:, 3] = NX.ravel()
    verts[:, 4] = NY.ravel()
    verts[:, 5] = NZ.ravel()
    verts[:, 6:9] = col.reshape(n, 3)
    rows, cols = H.shape
    base = np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]
    b = base.ravel()
    idx = np.concatenate([np.stack([b, b + 1, b + cols], axis=1),
                          np.stack([b + 1, b + cols + 1, b + cols], axis=1)]).ravel()
    return verts, idx.astype(np.uint32)


def _box_raw(sx: float, sy: float, sz: float) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    p = np.array([
        [[hx, hy, hz], [hx, -hy, hz], [hx, -hy, -hz]], [[hx, hy, hz], [hx, -hy, -hz], [hx, hy, -hz]],
        [[-hx, hy, -hz], [-hx, -hy, -hz], [-hx, -hy, hz]], [[-hx, hy, -hz], [-hx, -hy, hz], [-hx, hy, hz]],
        [[-hx, hy, hz], [-hx, -hy, hz], [hx, -hy, hz]], [[-hx, hy, hz], [hx, -hy, hz], [hx, hy, hz]],
        [[hx, hy, -hz], [hx, -hy, -hz], [-hx, -hy, -hz]], [[hx, hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz]],
        [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]], [[-hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]],
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz]], [[-hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]],
    ], dtype=np.float64)
    nrm = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
                    [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    return p.reshape(36, 3), np.repeat(nrm, 6, axis=0)


def _cone_raw(r: float, h: float, nseg: int) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    ring = np.stack([np.cos(ang) * r, np.zeros(nseg), np.sin(ang) * r], axis=1)
    apex = np.array([0.0, h, 0.0])
    tris, nrms = [], []
    for i in range(nseg):
        a, b = ring[i], ring[(i + 1) % nseg]
        tris.append([a, b, apex])
        m = (a + b) / 2.0
        side = np.array([m[0], 0.15 * r, m[2]])
        nrms.append(side / (np.linalg.norm(side) + 1e-9))
    return np.array(tris, dtype=np.float64).reshape(-1, 3), np.repeat(np.array(nrms), 3, axis=0)


def _xform(pos, nrm, x, y, z, ry, s, color):
    c, si = math.cos(ry), math.sin(ry)
    rot = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
    p = pos * s @ rot.T + np.array([x, y, z])
    col = np.tile(np.array(color, dtype=np.float64), (len(p), 1))
    return p, nrm @ rot.T, col


def _merge(parts):
    return (np.concatenate([a[0] for a in parts]), np.concatenate([a[1] for a in parts]),
            np.concatenate([a[2] for a in parts]))


def tree_verts(rng, x, z, y):
    parts = []
    ry = rng.uniform(0, 2 * math.pi)
    tint = rng.uniform(0.8, 1.15)
    scale = rng.uniform(0.85, 1.3)
    trunk_c = (0.35, 0.26, 0.16)
    leaf_c = (0.13 * tint, 0.34 * tint, 0.15 * tint)
    bp, bn = _box_raw(0.5, 2.4, 0.5)
    bp[:, 1] += 1.2
    parts.append(_xform(bp, bn, x, y, z, ry, scale, trunk_c))
    cp, cn = _cone_raw(1.7, 2.2, 6)
    cp[:, 1] += 2.2
    parts.append(_xform(cp, cn, x, y, z, ry, scale, leaf_c))
    cp, cn = _cone_raw(1.25, 1.8, 6)
    cp[:, 1] += 3.6
    parts.append(_xform(cp, cn, x, y, z, ry, scale, leaf_c))
    p, n, c = _merge(parts)
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


def bush_verts(rng, x, z, y, berries):
    parts = []
    ry = rng.uniform(0, 2 * math.pi)
    s = rng.uniform(0.8, 1.25)
    bp, bn = _box_raw(1.5, 1.1, 1.5)
    bp[:, 1] += 0.55
    parts.append(_xform(bp, bn, x, y, z, ry, s, (0.20, 0.38, 0.16)))
    if berries:
        for _ in range(5):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.3, 0.6)
            op, on = _box_raw(0.18, 0.18, 0.18)
            op[:, 1] += 0.85
            parts.append(_xform(op, on, x + math.cos(a) * r * s, y,
                                z + math.sin(a) * r * s, 0.0, s, (0.75, 0.12, 0.12)))
    p, n, c = _merge(parts)
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


def rock_verts(rng, x, z, y):
    s = rng.uniform(0.5, 1.1)
    ry = rng.uniform(0, 2 * math.pi)
    g = rng.uniform(0.38, 0.5)
    bp, bn = _box_raw(1.1, 0.8, 1.1)
    bp[:, 1] += 0.4 * s
    p, n, c = _xform(bp, bn, x, y, z, ry, s, (g, g, g * 1.04))
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


def gen_props(world: World, cx: int, cz: int) -> list[dict]:
    rng = np.random.default_rng(
        (WORLD_SEED * 73856093 ^ cx * 19349663 ^ cz * 83492791) & 0xFFFFFFFF)
    props = []
    for _ in range(34):
        x = rng.uniform(cx * CHUNK + 1, (cx + 1) * CHUNK - 1)
        z = rng.uniform(cz * CHUNK + 1, (cz + 1) * CHUNK - 1)
        fi, zi = int(np.clip(x, 0, WORLD - 1)), int(np.clip(z, 0, WORLD - 1))
        y = world.h[fi, zi]
        if y < WATER_LEVEL + 0.8 or y > 24 or world.ny[fi, zi] < 0.82:
            continue
        if world.forest[fi, zi] <= 0.5 and rng.random() > 0.18:
            continue
        r = rng.random()
        if r < 0.62:
            props.append({"type": "tree", "x": x, "y": y, "z": z, "r": 0.5,
                          "hp": 3, "alive": True, "rng": rng})
        elif r < 0.82:
            props.append({"type": "rock", "x": x, "y": y, "z": z, "r": 0.75,
                          "hp": 3, "alive": True, "rng": rng})
        elif r < 0.94:
            props.append({"type": "bush", "x": x, "y": y, "z": z, "r": 0.2,
                          "hp": 1, "berries": int(rng.integers(2, 5)), "alive": True, "rng": rng})
        else:
            props.append({"type": "bush", "x": x, "y": y, "z": z, "r": 0.2,
                          "hp": 1, "berries": 0, "alive": True, "rng": rng})
    return props


def build_prop_verts(props) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps, ns, cs = [], [], []
    for pr in props:
        if not pr["alive"]:
            continue
        if pr["type"] == "tree":
            p, n, c = tree_verts(pr["rng"], pr["x"], pr["z"], pr["y"])
        elif pr["type"] == "rock":
            p, n, c = rock_verts(pr["rng"], pr["x"], pr["z"], pr["y"])
        else:
            p, n, c = bush_verts(pr["rng"], pr["x"], pr["z"], pr["y"], pr.get("berries", 0) > 0)
        ps.append(p)
        ns.append(n)
        cs.append(c)
    if not ps:
        z0 = np.zeros((0, 3), dtype=np.float32)
        return z0, z0.copy(), z0.copy()
    return np.concatenate(ps), np.concatenate(ns), np.concatenate(cs)


def meat_drop_verts(x, y, z, raw=True):
    col = (0.75, 0.2, 0.2) if raw else (0.55, 0.3, 0.12)
    bp, bn = _box_raw(0.35, 0.2, 0.5)
    bp[:, 1] += 0.12
    p, n, c = _xform(bp, bn, x, y, z, 0.7, 1.0, col)
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


def fire_mesh(x, y, z):
    parts = []
    for i in range(5):
        a = i / 5 * 2 * math.pi
        sp, sn = _box_raw(0.22, 0.2, 0.22)
        sp[:, 1] += 0.1
        parts.append(_xform(sp, sn, x + math.cos(a) * 0.75, y, z + math.sin(a) * 0.75,
                            a, 1.0, (0.45, 0.44, 0.43)))
    for i in range(3):
        a = i / 3 * math.pi
        lp, ln = _box_raw(0.18, 0.18, 1.1)
        lp[:, 1] += 0.12
        parts.append(_xform(lp, ln, x, y, z, a, 1.0, (0.33, 0.2, 0.1)))
    p, n, c = _merge(parts)
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


def wolf_mesh():
    parts = []
    bp, bn = _box_raw(0.55, 0.5, 1.3)
    bp[:, 1] += 0.62
    parts.append(_xform(bp, bn, 0, 0, 0, 0, 1, (0.30, 0.29, 0.30)))
    hp, hn = _box_raw(0.36, 0.34, 0.42)
    hp[:, 1] += 0.95
    hp[:, 2] += 0.75
    parts.append(_xform(hp, hn, 0, 0, 0, 0, 1, (0.34, 0.32, 0.33)))
    for lx, lz in [(-0.2, 0.45), (0.2, 0.45), (-0.2, -0.45), (0.2, -0.45)]:
        lp, ln = _box_raw(0.14, 0.5, 0.14)
        lp[:, 1] += 0.25
        parts.append(_xform(lp, ln, lx, 0, lz, 0, 1, (0.26, 0.25, 0.26)))
    tp, tn = _box_raw(0.12, 0.12, 0.5)
    tp[:, 1] += 0.95
    tp[:, 2] -= 0.85
    parts.append(_xform(tp, tn, 0, 0, 0, 0, 1, (0.30, 0.29, 0.30)))
    p, n, c = _merge(parts)
    return p.astype(np.float32), n.astype(np.float32), c.astype(np.float32)


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


# ------------------------------------------------------------ игровая логика


def new_state(seed: int = WORLD_SEED) -> dict:
    world = World(seed)
    chunks = {}
    for cx in range(WORLD_CHUNKS):
        for cz in range(WORLD_CHUNKS):
            chunks[(cx, cz)] = {"props": gen_props(world, cx, cz)}
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
    return {
        "seed": seed, "world": world, "chunks": chunks,
        "px": sx, "py": world.height(sx, sz) + EYE, "pz": sz,
        "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.7, "pitch": -0.08,
        "on_ground": True,
        "hp": 100.0, "hunger": 85.0, "thirst": 80.0,
        "inv": {"дерево": 0, "камень": 0, "ягоды": 0, "сырое мясо": 0, "жареное мясо": 0},
        "tools": {"топор": False, "копьё": False},
        "fires": [], "wolves": [], "pickups": [],
        "time": DAY_LEN * 0.25, "days": 1, "attack_cd": 0.0,
        "messages": [], "craft_open": False, "dead": False,
        "spawn_pending": 3.0, "started": False, "show_help": False,
        "cold": False, "was_night": False,
    }


def message(st: dict, text: str):
    st["messages"] = [(text, 3.0)] + st["messages"][:3]


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
                rr = pr["r"] + PLAYER_R
                if 1e-6 < d2 < rr * rr:
                    d = math.sqrt(d2)
                    ox += ddx / d * (rr - d)
                    oz += ddz / d * (rr - d)
    return float(np.clip(ox, 2.0, WORLD - 2.0)), float(np.clip(oz, 2.0, WORLD - 2.0))


def update_player(st: dict, dt: float, move: tuple[float, float], sprint: bool, jump: bool):
    if st["dead"]:
        return
    yaw = st["yaw"]
    fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    wish = fwd * move[0] + right * move[1]
    wl = float(np.linalg.norm(wish))
    if wl > 1e-6:
        wish /= wl
    speed = SPRINT if (sprint and move[0] > 0) else WALK
    k = min(1.0, 14.0 * dt)
    st["vx"] += (wish[0] * speed - st["vx"]) * k
    st["vz"] += (wish[2] * speed - st["vz"]) * k
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
    if wl > 1e-6 and st["on_ground"]:
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
            ang = st["yaw"] + math.pi + float(np.random.default_rng().uniform(-1.2, 1.2))
            d = float(np.random.default_rng().uniform(28, 44))
            wx = float(np.clip(st["px"] + math.sin(ang) * d, 3, WORLD - 3))
            wz = float(np.clip(st["pz"] + math.cos(ang) * d, 3, WORLD - 3))
            st["wolves"].append({"x": wx, "z": wz, "yaw": ang, "hp": 3, "goal": ang,
                                 "wt": 0.0, "cd": 0.0, "hit_t": 0.0})
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
        elif night or w["hp"] < 3:
            sp = 1.6
            w["x"] += dx / dist * sp * dt
            w["z"] += dz / dist * sp * dt
        else:
            w["wt"] -= dt
            if w["wt"] <= 0:
                w["wt"] = float(np.random.default_rng().uniform(2.0, 5.0))
                w["goal"] = float(np.random.default_rng().uniform(0, 2 * math.pi))
            w["x"] += math.cos(w["goal"]) * 1.6 * dt
            w["z"] += math.sin(w["goal"]) * 1.6 * dt
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
        if w["hp"] <= 0:
            for _ in range(2):
                st["pickups"].append({"x": w["x"], "y": w["y"], "z": w["z"], "item": "сырое мясо"})
            message(st, "ВОЛК ПОВЕРЖЕН! +2 СЫРОГО МЯСА")
            continue
        alive.append(w)
    st["wolves"] = alive


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
                    if not pr["alive"]:
                        continue
                    if pr["type"] == "bush" and pr.get("berries", 0) == 0:
                        continue
                    dvec = np.array([pr["x"] - px, pr["y"] + 1.2 - py, pr["z"] - pz])
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
    if obj["type"] == "bush":
        n = obj["berries"]
        inv["ягоды"] += n
        obj["berries"] = 0
        message(st, f"+{n} ЯГОД")
        return (int(obj["x"] // CHUNK), int(obj["z"] // CHUNK))
    obj["hp"] -= 1
    if obj["type"] == "tree":
        n = 3 if st["tools"]["топор"] else 1
        inv["дерево"] += n
        message(st, "+1 ДЕРЕВО" if n == 1 else f"+{n} ДЕРЕВА")
    else:
        inv["камень"] += 2
        message(st, "+2 КАМНЯ")
    if obj["hp"] <= 0:
        obj["alive"] = False
        return (int(obj["x"] // CHUNK), int(obj["z"] // CHUNK))
    return None


def do_attack(st: dict) -> bool:
    if st["dead"] or st["attack_cd"] > 0:
        return False
    st["attack_cd"] = 0.55
    yaw = st["yaw"]
    fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    reach = 2.9 if st["tools"]["копьё"] else 2.2
    dmg = 2 if st["tools"]["копьё"] else 1
    for w in st["wolves"]:
        dx, dz = w["x"] - st["px"], w["z"] - st["pz"]
        d = math.hypot(dx, dz) + 1e-6
        if d < reach and (dx / d) * fwd[0] + (dz / d) * fwd[2] > 0.5:
            w["hp"] -= dmg
            w["hit_t"] = 0.25
            w["x"] += dx / d * 1.2
            w["z"] += dz / d * 1.2
            return None
    tgt = interact_target(st)
    if tgt and tgt[0] == "prop" and tgt[1]["type"] in ("tree", "rock"):
        return do_interact(st)
    return None


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
        message(st, "СОЗДАН ТОПОР! РУБКА БЫСТРЕЕ")
    elif name == "КОПЬЁ":
        st["tools"]["копьё"] = True
        message(st, "СОЗДАНО КОПЬЁ! УРОН x2")
    else:
        yaw = st["yaw"]
        fx = float(np.clip(st["px"] - math.sin(yaw) * 2.0, 2, WORLD - 2))
        fz = float(np.clip(st["pz"] - math.cos(yaw) * 2.0, 2, WORLD - 2))
        st["fires"].append({"x": fx, "z": fz, "y": st["world"].height(fx, fz), "fuel": 130.0})
        message(st, "КОСТЁР ГОРИТ! ТЕПЛО У ОГНЯ. E - ЖАРИТЬ МЯСО")
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
    st["days"] = 1 + int(st["time"] // DAY_LEN) if st["time"] >= (st["days"]) * DAY_LEN else st["days"]
    if after_night and not before_night:
        message(st, "НОЧЬ. ВОЛКИ ГОЛОДНЫ. ЖГИ КОСТЁР!")


def dump_glyphs(s: str):
    rows_out = ["" for _ in range(GLYPH_H)]
    for ch in s:
        g = _G[ch]
        for i in range(GLYPH_H):
            rows_out[i] += " " + g[i] + " "
    for r in rows_out:
        print("      " + r.rstrip())
    print()


def selftest() -> int:
    print("[1/6] пиксельный шрифт и атлас")
    atlas, index = build_font_atlas()
    assert atlas.dtype == np.uint8 and atlas.max() == 255
    missing = {ch for grp in UI_STRINGS for ch in grp if ch not in index}
    assert not missing, f"нет глифов: {missing}"
    for grp in UI_STRINGS:
        for ch in grp:
            g = _G[ch]
            assert len(g) == GLYPH_H and all(len(r) == GLYPH_W for r in g)
    print(f"      атлас {atlas.shape}, глифов {len(index)}; алфавит:")
    order = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    for i in range(0, len(order), 11):
        dump_glyphs(order[i:i + 11])

    print("[2/6] шум и рельеф")
    w = World()
    h = w.h
    assert np.all(np.isfinite(h)) and 0 <= h.min() and h.max() <= 34
    assert h.std() > 1.0, "рельеф плоский?"
    water_p = float((h < WATER_LEVEL).mean())
    assert water_p > 0.02, "нужны озёра"
    print(f"      высоты: мин {h.min():.1f}, макс {h.max():.1f}; под водой {water_p * 100:.0f}%")

    print("[3/6] чанки, меши, пропсы")
    va, ia = build_terrain_vertices(w, 3, 3)
    assert va.dtype == np.float32 and va.shape == ((CHUNK + 1) ** 2, 9)
    assert ia.dtype == np.uint32 and ia.shape[0] == CHUNK * CHUNK * 6
    assert np.all(np.isfinite(va))
    total = sum(len(gen_props(w, cx, cz)) for cx in range(WORLD_CHUNKS)
                for cz in range(WORLD_CHUNKS))
    props = gen_props(w, 3, 3)
    pv, pn, pc = build_prop_verts(props)
    assert pv.dtype == np.float32 and len(pv) == len(pn) == len(pc) and len(pv) > 0
    wm = wolf_mesh()
    assert len(wm[0]) == len(wm[1]) == len(wm[2]) and len(wm[0]) > 0
    fm = fire_mesh(10, 5, 10)
    assert len(fm[0]) == len(fm[1]) == len(fm[2])
    print(f"      меш чанка: {va.shape[0]} вершин / {ia.shape[0] // 3} треугольников")
    print(f"      пропсов в мире: {total}; меш волка: {len(wm[0])} вершин")

    print("[4/6] матрицы, проекция, фрустум")
    aspect = W_SIZE[0] / W_SIZE[1]
    P = perspective(FOV, aspect, 0.1, 260.0)
    eye = np.array([128.0, 12.0, 128.0])
    V = look_at(eye, 0.0, 0.0)
    PVm = P @ V
    ndc = PVm @ np.append(eye + np.array([0, 0, -10.0]), 1.0)
    ndc /= ndc[3]
    assert abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4 and -1 < ndc[2] < 1
    behind = PVm @ np.append(eye + np.array([0, 0, 10.0]), 1.0)
    assert behind[3] < 0
    side = PVm @ np.append(eye + np.array([100.0, 0, -10.0]), 1.0)
    assert (side / side[3])[0] > 1.0
    planes = frustum_planes(PVm)
    assert sphere_in_frustum(planes, eye + np.array([0, 0, -30.0]), 5.0)
    assert not sphere_in_frustum(planes, eye + np.array([0, 0, 30.0]), 5.0)
    up = PVm @ np.append(eye + np.array([0, 40.0, -10.0]), 1.0)
    assert (up / up[3])[1] > 0
    O = ortho_px(1280, 720)
    assert np.allclose(O @ np.array([0, 0, 0, 1]), [-1, 1, 0, 1])
    assert np.allclose(O @ np.array([1280, 720, 0, 1]), [1, -1, 0, 1])
    print("      NDC-центр, отсечение позади, фрустум, орто в пикселях: ОК")

    print("[5/6] игровая логика: движение, добыча, крафт, еда")
    st = new_state()
    g0 = st["world"].height(st["px"], st["pz"])
    assert abs(st["py"] - (g0 + EYE)) < 1e-6
    update_player(st, 0.016, (1.0, 0.0), False, False)
    update_player(st, 0.016, (1.0, 0.0), False, False)
    assert st["hunger"] < 85.0 and st["thirst"] < 80.0
    for _ in range(400):
        update_player(st, 0.016, (0.0, 0.0), False, False)
    g = st["world"].height(st["px"], st["pz"])
    assert abs(st["py"] - (g + EYE)) < 1e-6, "игрок должен стоять на земле"
    st["inv"]["дерево"], st["inv"]["камень"] = 3, 2
    assert try_craft(st, 0) and st["tools"]["топор"]
    assert st["inv"]["дерево"] == 0 and st["inv"]["камень"] == 0
    assert not try_craft(st, 0)
    st["inv"]["дерево"], st["inv"]["камень"] = 2, 1
    assert try_craft(st, 1)
    st["inv"]["дерево"], st["inv"]["камень"] = 5, 4
    assert try_craft(st, 2)
    assert len(st["fires"]) == 1 and st["fires"][0]["fuel"] > 0
    st["inv"]["сырое мясо"] = 2
    f0 = st["fires"][0]
    st["px"], st["pz"] = f0["x"] + 0.8, f0["z"]
    assert cook_at_fire(st) and st["inv"]["жареное мясо"] == 1 and st["inv"]["сырое мясо"] == 1
    eat_food(st)
    assert st["hunger"] > 85.0 and st["inv"]["жареное мясо"] == 0
    bush = None
    for ch in st["chunks"].values():
        for p in ch["props"]:
            if p["type"] == "bush" and p["berries"] > 0:
                bush = p
                break
        if bush:
            break
    st["px"], st["pz"] = bush["x"] + 0.9, bush["z"]
    st["py"] = st["world"].height(st["px"], st["pz"]) + EYE
    st["yaw"] = math.atan2(-(bush["x"] - st["px"]), -(bush["z"] - st["pz"]))
    tgt = interact_target(st)
    assert tgt is not None and tgt[0] == "prop" and tgt[1] is bush, f"куст не нацелен: {tgt}"
    n_berries = bush["berries"]
    do_interact(st)
    assert st["inv"]["ягоды"] >= n_berries
    print(f"      куст +{n_berries} ягод")

    print("[6/6] волки, день/ночь, тепло")
    st2 = new_state()
    st2["time"] = DAY_LEN * 0.75
    assert sun_elev(st2) < -0.02, "ночью солнце под горизонтом"
    st2["spawn_pending"] = 0.0
    update_wolves(st2, 0.1)
    assert len(st2["wolves"]) == 1, "ночью должен появиться волк"
    wb = st2["wolves"][0]
    for _ in range(60):
        update_wolves(st2, 0.5)
    assert st2["hp"] < 100.0, "волк должен наносить урон"

    for _ in range(120):
        update_wolves(st2, 0.05)
    dist1 = math.hypot(wb["x"] - st2["px"], wb["z"] - st2["pz"])
    assert dist1 < 3.5, "волк должен держаться рядом в бою"
    st2b = new_state()
    st2b["time"] = DAY_LEN * 0.25
    update_wolves(st2b, 0.2)
    assert len(st2b["wolves"]) == 0, "днём волков быть не должно"
    wf = new_state()
    wf["time"] = DAY_LEN * 0.75
    wf["fires"] = [{"x": wf["px"], "z": wf["pz"], "y": 0.0, "fuel": 100.0}]
    for _ in range(120):
        update_survival(wf, 0.1)
    assert not wf["cold"] and wf["hp"] == 100.0, "у костра тепло"
    wc = new_state()
    wc["time"] = DAY_LEN * 0.75
    for _ in range(120):
        update_survival(wc, 0.1)
    assert wc["cold"] and wc["hp"] < 100.0, "ночью без огня холодно"
    print("      спавн/погоня/урон волков, день-ночь, тепло-холод: ОК")
    print()
    print("SELFTEST OK — логика, генерация мира и математика проверены")
    return 0
