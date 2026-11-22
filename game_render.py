"""Рендер: шейдеры и Graphic-пайплайн на moderngl."""
import math

import numpy as np
import moderngl as mgl

from game_core import (
    CHUNK, EYE as EYE_H, FOV, FOG_FAR, FOG_NEAR, VIEW_CHUNKS, WATER_LEVEL as WATER_Y, WORLD,
    build_font_atlas, text_width,
    build_terrain_vertices, build_prop_verts, wolf_mesh, fire_mesh, meat_drop_verts,
    perspective, look_at, ortho_px, frustum_planes, sphere_in_frustum, model_rot_y,
    can_craft, RECIPES, HELP_LINES, CELL_W as CELL_ADV,
)


def write_mat(prog, name, m: np.ndarray):
    prog[name].write(np.ascontiguousarray(m.T, dtype="f4").tobytes())


TERRAIN_VS = """
#version 330
uniform mat4 u_pv;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_world;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_color = in_color;
    vec3 n = mat3(u_model) * in_normal;
    v_normal = normalize(n);
    gl_Position = u_pv * w;
}
"""

TERRAIN_FS = """
#version 330
uniform vec3 u_sun_dir;
uniform vec3 u_light;
uniform float u_amb;
uniform vec3 u_cam;
uniform vec3 u_fog_color;
uniform vec2 u_fog;
uniform float u_alpha;
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world;
out vec4 out_color;
void main() {
    float lam = max(dot(normalize(v_normal), normalize(u_sun_dir)), 0.0);
    vec3 col = v_color * (u_amb + u_light * lam);
    float d = length(u_cam - v_world);
    float f = smoothstep(u_fog.x, u_fog.y, d);
    col = mix(col, u_fog_color, f);
    out_color = vec4(col, u_alpha);
}
"""

SKY_VS = """
#version 330
in vec2 in_ndc;
out vec2 v_ndc;
void main() {
    v_ndc = in_ndc;
    gl_Position = vec4(in_ndc, 0.9999, 1.0);
}
"""

SKY_FS = """
#version 330
uniform vec3 u_fwd;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_tan_fov;
uniform float u_aspect;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
in vec2 v_ndc;
out vec4 out_color;
void main() {
    vec3 dir = normalize(u_fwd + u_right * (v_ndc.x * u_tan_fov * u_aspect)
                                   + u_up * (v_ndc.y * u_tan_fov));
    float t = clamp(dir.y * 1.4 + 0.08, 0.0, 1.0);
    vec3 col = mix(u_horizon, u_zenith, pow(t, 0.6));
    float s = max(dot(dir, normalize(u_sun_dir)), 0.0);
    col += u_sun_color * (pow(s, 900.0) * 1.2 + pow(s, 24.0) * 0.18);
    out_color = vec4(col, 1.0);
}
"""

HUD_VS = """
#version 330
uniform mat4 u_ortho;
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_col;
out vec2 v_uv;
out vec4 v_col;
void main() {
    v_uv = in_uv;
    v_col = in_col;
    gl_Position = u_ortho * vec4(in_pos, 0.0, 1.0);
}
"""

HUD_FS = """
#version 330
uniform sampler2D u_tex;
uniform int u_use_tex;
in vec2 v_uv;
in vec4 v_col;
out vec4 out_color;
void main() {
    float a = 1.0;
    if (u_use_tex == 1) {
        a = texture(u_tex, v_uv).r;
        if (a < 0.5) discard;
    }
    out_color = vec4(v_col.rgb, v_col.a * a);
}
"""

FLAT_VS = """
#version 330
uniform mat4 u_pv;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_pv * vec4(in_pos, 1.0);
}
"""

FLAT_FS = """
#version 330
in vec3 v_color;
out vec4 out_color;
void main() {
    out_color = vec4(v_color, 1.0);
}
"""


class Renderer:
    def __init__(self, ctx, size: tuple[int, int]):
        self.ctx = ctx
        self.w, self.h = size
        self.proj = perspective(FOV, self.w / self.h, 0.1, 260.0)
        self.ortho = ortho_px(self.w, self.h)
        self.prog = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        self.flat_prog = ctx.program(vertex_shader=FLAT_VS, fragment_shader=FLAT_FS)
        atlas, self.font_index = build_font_atlas()
        self.atlas_w, self.atlas_h = atlas.shape[1], atlas.shape[0]
        self.font_tex = ctx.texture((self.atlas_w, self.atlas_h), 1, atlas.tobytes(), dtype="f1")
        self.font_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self.font_tex.repeat_x = False
        self.font_tex.repeat_y = False
        self.hud_prog["u_tex"].value = 0
        corners = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        sky_vbo = ctx.buffer(corners.tobytes())
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(sky_vbo, "2f", "in_ndc")])
        self.chunk_terrain = {}
        self.chunk_props = {}
        wm_p, wm_n, wm_c = wolf_mesh()
        self.wolf_vbo = ctx.buffer(np.concatenate([wm_p, wm_n, wm_c], axis=1).tobytes())
        self.wolf_vao = ctx.vertex_array(
            self.prog, [(self.wolf_vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
        self.hud_buf_solid = ctx.buffer(reserve=1024)
        self.hud_buf_text = ctx.buffer(reserve=1024)
        self.hud_vao_solid = ctx.vertex_array(
            self.hud_prog, [(self.hud_buf_solid, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.hud_vao_text = ctx.vertex_array(
            self.hud_prog, [(self.hud_buf_text, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.flat_buf = ctx.buffer(reserve=1024)
        self.flat_vao = ctx.vertex_array(
            self.flat_prog, [(self.flat_buf, "3f 3f", "in_pos", "in_color")])
        self.hud_solid = []
        self.hud_text = []
        self.water_res = None

    def build_world_gpu(self, st: dict):
        world = st["world"]
        for key in self.chunk_terrain:
            vao, vbo, ibo = self.chunk_terrain[key]
            vao.release()
            vbo.release()
            ibo.release()
        for key in self.chunk_props:
            vao, bufs = self.chunk_props[key]
            if vao is not None:
                vao.release()
            for b in bufs:
                b.release()
        self.chunk_terrain.clear()
        self.chunk_props.clear()
        for cx in range(WORLD // CHUNK):
            for cz in range(WORLD // CHUNK):
                verts, idx = build_terrain_vertices(world, cx, cz)
                vbo = self.ctx.buffer(verts.tobytes())
                ibo = self.ctx.buffer(idx.tobytes())
                vao = self.ctx.vertex_array(
                    self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
                    index_buffer=ibo, index_element_size=4)
                self.chunk_terrain[(cx, cz)] = (vao, vbo, ibo)
                self.rebuild_props_gpu((cx, cz), st["chunks"][(cx, cz)]["props"])

    def rebuild_props_gpu(self, key, props):
        old = self.chunk_props.pop(key, None)
        if old is not None:
            if old[0] is not None:
                old[0].release()
            for b in old[1]:
                b.release()
        p, n, c = build_prop_verts(props)
        if len(p) == 0:
            self.chunk_props[key] = (None, [])
            return
        vbo = self.ctx.buffer(np.concatenate([p, n, c], axis=1).tobytes())
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
        self.chunk_props[key] = (vao, [vbo])

    def _basis(self, yaw: float, pitch: float) -> tuple[np.ndarray, ...]:
        fwd = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                        -math.cos(yaw) * math.cos(pitch)])
        f = fwd / np.linalg.norm(fwd)
        right = np.cross(f, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return f, right, up

    def draw_world(self, st: dict, sun: dict):
        ctx = self.ctx
        eye = np.array([st["px"], st["py"], st["pz"]])
        self.view = look_at(eye, st["yaw"], st["pitch"])
        pv = self.proj @ self.view
        planes = frustum_planes(pv)

        ctx.clear(*sun["sky"])
        f, right, up = self._basis(st["yaw"], st["pitch"])
        self.sky_prog["u_fwd"].value = tuple(f)
        self.sky_prog["u_right"].value = tuple(right)
        self.sky_prog["u_up"].value = tuple(up)
        self.sky_prog["u_tan_fov"].value = math.tan(math.radians(FOV) / 2.0)
        self.sky_prog["u_aspect"].value = self.w / self.h
        self.sky_prog["u_zenith"].value = tuple(np.clip(sun["sky"] * 1.15, 0, 1))
        self.sky_prog["u_horizon"].value = tuple(np.clip(sun["sky"] * 0.25, 0, 1))
        self.sky_prog["u_sun_dir"].value = tuple(sun["sun_dir"])
        self.sky_prog["u_sun_color"].value = tuple(np.clip(sun["light"] * 1.3, 0, 1))
        ctx.disable(mgl.DEPTH_TEST)
        self.sky_vao.render(mgl.TRIANGLES)
        ctx.enable(mgl.DEPTH_TEST)

        write_mat(self.prog, "u_pv", pv)
        write_mat(self.prog, "u_model", np.eye(4))
        self.prog["u_sun_dir"].value = tuple(sun["sun_dir"])
        self.prog["u_light"].value = tuple(sun["light"])
        self.prog["u_amb"].value = sun["amb"]
        self.prog["u_cam"].value = (st["px"], st["py"], st["pz"])
        self.prog["u_fog_color"].value = tuple(sun["sky"])
        self.prog["u_fog"].value = (FOG_NEAR, FOG_FAR)
        self.prog["u_alpha"].value = 1.0

        visible = []
        for key, (vao, _vbo, _ibo) in self.chunk_terrain.items():
            ccx, ccz = key[0] * CHUNK + CHUNK / 2, key[1] * CHUNK + CHUNK / 2
            d = math.hypot(ccx - st["px"], ccz - st["pz"])
            if d > (VIEW_CHUNKS + 0.8) * CHUNK:
                continue
            if not sphere_in_frustum(planes, np.array([ccx, 14.0, ccz]), CHUNK * 0.75 + 10.0):
                continue
            visible.append(key)
            vao.render(mgl.TRIANGLES)
        for key in visible:
            vao = self.chunk_props.get(key, (None,))[0]
            if vao is not None:
                vao.render(mgl.TRIANGLES)

        for w in st["wolves"]:
            if math.hypot(w["x"] - st["px"], w["z"] - st["pz"]) > FOG_FAR:
                continue
            write_mat(self.prog, "u_model", model_rot_y(w["yaw"] + math.pi, w["x"], w["y"], w["z"]))
            self.wolf_vao.render(mgl.TRIANGLES)
        write_mat(self.prog, "u_model", np.eye(4))

        misc = []
        for pk in st["pickups"]:
            misc.append(np.concatenate(meat_drop_verts(pk["x"], pk["y"], pk["z"],
                                                       raw=pk["item"] == "сырое мясо"), axis=1))
        for fr in st["fires"]:
            misc.append(np.concatenate(fire_mesh(fr["x"], fr["y"], fr["z"]), axis=1))
        if misc:
            data = np.concatenate(misc).astype("f4")
            buf = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(buf, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
            vao.render(mgl.TRIANGLES)
            buf.release()
            vao.release()

        flame = self.build_flame(st)
        if flame:
            arr = np.asarray(flame, dtype="f4")
            self.flat_buf.orphan(arr.nbytes)
            self.flat_buf.write(arr.tobytes())
            write_mat(self.flat_prog, "u_pv", pv)
            self.flat_vao.render(mgl.TRIANGLES)

        if self.water_res is None:
            cc = (0.13, 0.30, 0.47)
            quad = np.array([
                [0, WATER_Y, 0, 0, 1, 0, *cc], [WORLD, WATER_Y, 0, 0, 1, 0, *cc],
                [WORLD, WATER_Y, WORLD, 0, 1, 0, *cc],
                [0, WATER_Y, 0, 0, 1, 0, *cc], [WORLD, WATER_Y, WORLD, 0, 1, 0, *cc],
                [0, WATER_Y, WORLD, 0, 1, 0, *cc],
            ], dtype="f4")
            vbo = self.ctx.buffer(quad.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")])
            self.water_res = (vao, vbo)
        if st["py"] - EYE_H < WATER_Y + 25.0:
            ctx.enable(mgl.BLEND)
            ctx.depth_mask = False
            self.prog["u_alpha"].value = 0.62
            self.water_res[0].render(mgl.TRIANGLES)
            self.prog["u_alpha"].value = 1.0
            ctx.depth_mask = True
            ctx.disable(mgl.BLEND)

    def build_flame(self, st: dict) -> list:
        out = []
        t = st["time"]
        for fr in st["fires"]:
            sway = math.sin(t * 7 + fr["x"]) * 0.06
            hh = 0.55 + 0.14 * math.sin(t * 11 + fr["z"])
            ang0 = t * 3.0
            apex = (fr["x"] + sway, fr["y"] + 0.2 + hh, fr["z"])
            ring = [(fr["x"] + math.cos(ang0 + i / 5 * 2 * math.pi) * 0.22,
                     fr["y"] + 0.18,
                     fr["z"] + math.sin(ang0 + i / 5 * 2 * math.pi) * 0.22) for i in range(5)]
            col = (1.0, 0.6, 0.12)
            for i in range(5):
                a, b = ring[i], ring[(i + 1) % 5]
                for pt in (a, b, apex):
                    out.extend([pt[0], pt[1], pt[2], col[0], col[1], col[2]])
        return out

    # ------------------------------------------------------------------ HUD

    def rect(self, x, y, w, h, col):
        self.hud_solid.extend([
            x, y, 0, 0, *col, x + w, y, 0, 0, *col, x, y + h, 0, 0, *col,
            x + w, y, 0, 0, *col, x + w, y + h, 0, 0, *col, x, y + h, 0, 0, *col,
        ])

    def text(self, x, y, s, col=(1, 1, 1, 1), scale=2):
        s = s.upper()
        aw, ah = self.atlas_w, self.atlas_h
        for ch in s:
            cell = self.font_index.get(ch)
            if cell is not None:
                gx0, gy0 = cell
                u0, u1 = gx0 / aw, (gx0 + 10) / aw
                v0, v1 = gy0 / ah, (gy0 + 14) / ah
                x0, y0, x1, y1 = x, y, x + 10 * scale, y + 14 * scale
                self.hud_text.extend([
                    x0, y0, u0, v0, *col, x1, y0, u1, v0, *col, x0, y1, u0, v1, *col,
                    x1, y0, u1, v0, *col, x1, y1, u1, v1, *col, x0, y1, u0, v1, *col,
                ])
            x += CELL_ADV * scale

    def centered(self, y, s, col=(1, 1, 1, 1), scale=2):
        self.text(self.w / 2 - text_width(s, scale) / 2, y, s, col, scale)

    def flush_hud(self):
        ctx = self.ctx
        ctx.disable(mgl.DEPTH_TEST)
        ctx.enable(mgl.BLEND)
        write_mat(self.hud_prog, "u_ortho", self.ortho)
        self.hud_prog["u_use_tex"].value = 0
        if self.hud_solid:
            arr = np.asarray(self.hud_solid, dtype="f4")
            self.hud_buf_solid.orphan(arr.nbytes)
            self.hud_buf_solid.write(arr.tobytes())
            self.hud_vao_solid.render(mgl.TRIANGLES)
            self.hud_solid = []
        self.font_tex.use(0)
        self.hud_prog["u_use_tex"].value = 1
        if self.hud_text:
            arr = np.asarray(self.hud_text, dtype="f4")
            self.hud_buf_text.orphan(arr.nbytes)
            self.hud_buf_text.write(arr.tobytes())
            self.hud_vao_text.render(mgl.TRIANGLES)
            self.hud_text = []
        ctx.disable(mgl.BLEND)
        ctx.enable(mgl.DEPTH_TEST)

    def bar(self, x, y, label, frac, col):
        frac = float(np.clip(frac, 0.0, 1.0))
        self.text(x, y, label, (1, 1, 1, 0.92))
        self.rect(x + 205, y + 3, 170, 12, (0, 0, 0, 0.55))
        if frac > 0.005:
            self.rect(x + 207, y + 5, 166 * frac, 8, (*col, 0.95))

    def draw_hud(self, st: dict, sun: dict, fps: float):
        cfg = [
            ("ЗДОРОВЬЕ", st["hp"] / 100, (0.85, 0.15, 0.12)),
            ("ГОЛОД", st["hunger"] / 100, (0.93, 0.55, 0.10)),
            ("ЖАЖДА", st["thirst"] / 100, (0.15, 0.5, 0.9)),
        ]
        y = self.h - 96
        for label, frac, col in cfg:
            self.bar(16, y, label, frac, col)
            y += 26
        self.bar(16, y, "ТЕПЛО", 0.15 if st["cold"] else 1.0, (0.9, 0.75, 0.2))
        hh, mm = int(sun["hour"]), int((sun["hour"] % 1) * 60)
        self.text(16, 14, f"{hh:02d}:{mm:02d}  "
                  + ("ДЕНЬ" if sun["day"] else "НОЧЬ") + "  ДЕНЬ " + str(st["days"]),
                  (1, 1, 1, 0.92))
        self.text(16, 34, "FPS " + str(int(round(fps))), (1, 1, 1, 0.6))

        inv = st["inv"]
        tools = [n for n, has in (("ТОПОР", st["tools"]["топор"]),
                                  ("КОПЬЁ", st["tools"]["копьё"])) if has]
        counts = [
            ("ДЕРЕВО " + str(inv["дерево"]), inv["дерево"] > 0),
            ("КАМЕНЬ " + str(inv["камень"]), inv["камень"] > 0),
            ("ЯГОДЫ " + str(inv["ягоды"]), inv["ягоды"] > 0),
            ("СЫРОЕ МЯСО " + str(inv["сырое мясо"]), inv["сырое мясо"] > 0),
            ("ЖАРЕНОЕ МЯСО " + str(inv["жареное мясо"]), inv["жареное мясо"] > 0),
        ]
        yy = 14
        for line in tools:
            self.text_r(line, yy, (1, 0.9, 0.4, 0.95))
            yy += 20
        for line, has in counts:
            if has:
                self.text_r(line, yy, (1, 1, 1, 0.9))
                yy += 20

        cx, cy = self.w / 2, self.h / 2
        self.rect(cx - 9, cy - 1, 7, 2, (1, 1, 1, 0.85))
        self.rect(cx + 2, cy - 1, 7, 2, (1, 1, 1, 0.85))
        self.rect(cx - 1, cy - 9, 2, 7, (1, 1, 1, 0.85))
        self.rect(cx - 1, cy + 2, 2, 7, (1, 1, 1, 0.85))

        my = self.h * 0.30
        for text, ttl in st["messages"][:4]:
            a = float(np.clip(ttl, 0.0, 1.0))
            self.centered(my, text, (1, 1, 0.8, a))
            my += 22
        if st["cold"] and not st["dead"]:
            self.centered(self.h - 30, "ХОЛОД! РАЗВЕДИ КОСТЁР (C)", (0.5, 0.8, 1, 0.95))
        if st["craft_open"]:
            self.rect(self.w / 2 - 225, self.h / 2 - 108, 450, 206, (0, 0, 0, 0.72))
            self.text(self.w / 2 - 205, self.h / 2 - 94, "КРАФТ:", (1, 0.95, 0.6, 1))
            yy = self.h / 2 - 66
            for i, (name, cost) in enumerate(RECIPES):
                line = f"{i + 1} - {name}: " + " + ".join(f"{v} {k.upper()}" for k, v in cost)
                self.text(self.w / 2 - 205, yy, line,
                          (0.55, 1, 0.55, 0.95) if can_craft(st, i) else (0.7, 0.7, 0.7, 0.75))
                yy += 22
            self.text(self.w / 2 - 205, yy + 4, "1/2/3 - СДЕЛАТЬ, C - ЗАКРЫТЬ", (1, 1, 1, 0.7))
        if st["show_help"] and st["started"] and not st["dead"]:
            self.rect(self.w / 2 - 270, 64, 540, 130, (0, 0, 0, 0.6))
            yy = 78
            for ln in HELP_LINES:
                self.centered(yy, ln, (1, 1, 1, 0.92))
                yy += 20
            self.centered(yy + 4, "H - УБРАТЬ ПОМОЩЬ", (1, 0.9, 0.4, 0.8))
        if st["dead"]:
            self.rect(0, 0, self.w, self.h, (0.25, 0.0, 0.0, 0.55))
            self.centered(self.h / 2 - 70, "ВЫ ПОГИБЛИ", (1, 0.2, 0.15, 1), 6)
            self.centered(self.h / 2 + 16, f"ВЫЖИЛ ДНЕЙ: {st['days']}", (1, 1, 1, 0.95), 3)
            self.centered(self.h / 2 + 56, "ПРОБЕЛ - ЗАНОВО", (1, 1, 1, 0.85))
        elif not st["started"]:
            self.rect(0, 0, self.w, self.h, (0, 0, 0, 0.6))
            self.centered(self.h * 0.13, "ЛЕС. ВЫЖИВАНИЕ", (0.6, 1.0, 0.5, 1), 6)
            yy = self.h * 0.13 + 76
            for ln in HELP_LINES:
                self.centered(yy, ln, (1, 1, 1, 0.92))
                yy += 24
            self.centered(yy + 12, "ПРОБЕЛ - НАЧАТЬ", (1, 0.9, 0.4, 0.6 + 0.4 * math.sin(st["time"] * 5)))

    def text_r(self, s, y, col):
        self.text(self.w - 16 - text_width(s, 2), y, s, col, 2)
