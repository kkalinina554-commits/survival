"""
Рендер v2: шейдеры и GPU-пайплайн на moderngl.
Программы: небо, ландшафт (атлас 3 тайла по сплату), пропсы/животные (атлас по тайлу),
плоские текстуры (огонь/вода/подборы), HUD (шрифт).
"""
import math

import numpy as np
import moderngl as mgl

from game_core import (
    CHUNK, EYE as EYE_H, FOV, FOG_FAR, FOG_NEAR, VIEW_CHUNKS, WATER_LEVEL as WATER_Y, WORLD,
    ATLAS_COLS, ATLAS_ROWS, T_GRASS, T_ROCK, T_SAND, T_FLAME, T_RED, T_WATER,
    build_font_atlas, build_texture_atlas, atlas_tile_uv, text_width,
    build_terrain_chunk, build_prop_verts, wolf_model, boar_model, fire_model,
    _hand_item_set,
    perspective, look_at, ortho_px, frustum_planes, sphere_in_frustum,
    model_rot_y, model_translate, model_rot_x, mat4_mul,
    can_craft, RECIPES, HELP_LINES, HOTBAR, CELL_W as CELL_ADV,
)


def write_mat(prog, name, m: np.ndarray):
    u = prog.get(name, None)
    if u is not None:
        u.write(np.ascontiguousarray(m.T, dtype="f4").tobytes())


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

TERRAIN_VS = """
#version 330
uniform mat4 u_pv;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_splat;
in vec2 in_uv;
out vec3 v_normal;
out vec3 v_world;
out vec3 v_splat;
out vec2 v_uv;
void main() {
    v_world = in_pos;
    v_normal = in_normal;
    v_splat = in_splat;
    v_uv = in_uv;
    gl_Position = u_pv * vec4(in_pos, 1.0);
}
"""

TERRAIN_FS = """
#version 330
uniform sampler2D u_atlas;
uniform vec2 u_tf;
uniform vec2 u_off_grass;
uniform vec2 u_off_rock;
uniform vec2 u_off_sand;
uniform vec3 u_sun_dir;
uniform vec3 u_light;
uniform float u_amb;
uniform vec3 u_cam;
uniform vec3 u_fog_color;
uniform vec2 u_fog;
in vec3 v_normal;
in vec3 v_world;
in vec3 v_splat;
in vec2 v_uv;
out vec4 out_color;
void main() {
    vec2 texel = u_tf / 64.0;
    vec2 tuv = clamp(fract(v_uv) * u_tf, texel * 0.5, u_tf - texel * 0.5);
    vec3 g = texture(u_atlas, tuv + u_off_grass).rgb;
    vec3 r = texture(u_atlas, tuv + u_off_rock).rgb;
    vec3 s = texture(u_atlas, tuv + u_off_sand).rgb;
    vec3 col = v_splat.x * g + v_splat.y * r + v_splat.z * s;
    float lam = max(dot(normalize(v_normal), normalize(u_sun_dir)), 0.0);
    col *= (u_amb + u_light * lam);
    float d = length(u_cam - v_world);
    col = mix(col, u_fog_color, smoothstep(u_fog.x, u_fog.y, d));
    out_color = vec4(col, 1.0);
}
"""

PROP_VS = """
#version 330
uniform mat4 u_pv;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_tile;
out vec3 v_normal;
out vec3 v_world;
out vec2 v_uv;
out float v_tile;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    v_tile = in_tile;
    gl_Position = u_pv * w;
}
"""

PROP_FS = """
#version 330
uniform sampler2D u_atlas;
uniform vec2 u_tf;
uniform vec3 u_sun_dir;
uniform vec3 u_light;
uniform float u_amb;
uniform vec3 u_cam;
uniform vec3 u_fog_color;
uniform vec2 u_fog;
in vec3 v_normal;
in vec3 v_world;
in vec2 v_uv;
in float v_tile;
out vec4 out_color;
void main() {
    float tile = floor(v_tile + 0.5);
    vec2 off = vec2(mod(tile, 4.0), floor(tile / 4.0)) * u_tf;
    vec2 texel = u_tf / 64.0;
    vec2 tuv = clamp(fract(v_uv) * u_tf, texel * 0.5, u_tf - texel * 0.5);
    vec4 tex = texture(u_atlas, tuv + off);
    if (tex.a < 0.4) discard;
    float lam = max(dot(normalize(v_normal), normalize(u_sun_dir)), 0.0);
    vec3 col = tex.rgb * (u_amb + u_light * lam);
    float d = length(u_cam - v_world);
    col = mix(col, u_fog_color, smoothstep(u_fog.x, u_fog.y, d));
    out_color = vec4(col, 1.0);
}
"""

FLAT_TEX_VS = """
#version 330
uniform mat4 u_pv;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = u_pv * vec4(in_pos, 1.0);
}
"""

FLAT_TEX_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_tf;
uniform vec2 u_off;
uniform vec2 u_scroll;
uniform float u_alpha;
uniform vec3 u_mul;
in vec2 v_uv;
out vec4 out_color;
void main() {
    vec2 texel = u_tf / 64.0;
    vec2 tuv = clamp(fract(v_uv + u_scroll) * u_tf, texel * 0.5, u_tf - texel * 0.5);
    vec4 tex = texture(u_tex, tuv + u_off);
    if (tex.a < 0.05) discard;
    out_color = vec4(tex.rgb * u_mul, u_alpha * tex.a);
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


def _animal_gpu_res(ctx, prog, model):
    body_v, body_i, hips, legs = model
    body_vbo = ctx.buffer(body_v.tobytes())
    body_ibo = ctx.buffer(body_i.astype("u4").tobytes())
    body_vao = ctx.vertex_array(
        prog, [(body_vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_tile")],
        index_buffer=body_ibo, index_element_size=4)
    leg_res = []
    for lv, li in legs:
        lvbo = ctx.buffer(np.ascontiguousarray(lv).tobytes())
        libo = ctx.buffer(li.astype("u4").tobytes())
        lvao = ctx.vertex_array(
            prog, [(lvbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_tile")],
            index_buffer=libo, index_element_size=4)
        leg_res.append((lvao, lvbo, libo))
    return {"body": body_vao, "body_bufs": (body_vbo, body_ibo), "legs": leg_res, "hips": hips}


class Renderer:
    def __init__(self, ctx, size: tuple[int, int]):
        self.ctx = ctx
        self.w, self.h = size
        self.proj = perspective(FOV, self.w / self.h, 0.1, 260.0)
        self.ortho = ortho_px(self.w, self.h)
        self.prog = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.prop_prog = ctx.program(vertex_shader=PROP_VS, fragment_shader=PROP_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        self.ftex_prog = ctx.program(vertex_shader=FLAT_TEX_VS, fragment_shader=FLAT_TEX_FS)

        atlas, self.font_index = build_font_atlas()
        self.atlas_w, self.atlas_h = atlas.shape[1], atlas.shape[0]
        self.font_tex = ctx.texture((self.atlas_w, self.atlas_h), 1, atlas.tobytes(), dtype="f1")
        self.font_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self.hud_prog["u_tex"].value = 0

        rgba = build_texture_atlas()
        self.world_tex = ctx.texture((ATLAS_COLS * 64, ATLAS_ROWS * 64), 4, rgba.tobytes(), dtype="f1")
        self.world_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self.world_tex.repeat_x = True
        self.world_tex.repeat_y = True

        corners = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        sky_vbo = ctx.buffer(corners.tobytes())
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(sky_vbo, "2f", "in_ndc")])

        self.chunk_terrain = {}
        self.chunk_props = {}
        self.fire_res = self._make_indexed(fire_model()[0], fire_model()[1], self.prop_prog)
        self.wolf_res = _animal_gpu_res(ctx, self.prop_prog, wolf_model())
        self.boar_res = _animal_gpu_res(ctx, self.prop_prog, boar_model())
        self.hand_res = {k: self._make_indexed(v, i, self.prop_prog)
                         for k, (v, i) in _hand_item_set().items()}

        self.hud_buf_solid = ctx.buffer(reserve=1024)
        self.hud_buf_text = ctx.buffer(reserve=1024)
        self.hud_vao_solid = ctx.vertex_array(
            self.hud_prog, [(self.hud_buf_solid, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.hud_vao_text = ctx.vertex_array(
            self.hud_prog, [(self.hud_buf_text, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.flat_buf = ctx.buffer(reserve=1024)
        self.flat_vao = ctx.vertex_array(
            self.ftex_prog, [(self.flat_buf, "3f 2f", "in_pos", "in_uv")])
        self.hud_solid = []
        self.hud_text = []
        self.water_res = None

    def _make_indexed(self, v9: np.ndarray, idx: np.ndarray, prog):
        vbo = self.ctx.buffer(v9.tobytes())
        ibo = self.ctx.buffer(idx.astype("u4").tobytes())
        vao = self.ctx.vertex_array(
            prog, [(vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_tile")],
            index_buffer=ibo, index_element_size=4)
        return {"vao": vao, "bufs": (vbo, ibo)}

    def build_world_gpu(self, st: dict):
        world = st["world"]
        for key in self.chunk_terrain:
            vao, vbo, ibo = self.chunk_terrain[key]
            vao.release()
            vbo.release()
            ibo.release()
        self.chunk_terrain.clear()
        for key in self.chunk_props:
            vao, bufs = self.chunk_props[key]
            if vao is not None:
                vao.release()
            for b in bufs:
                b.release()
        self.chunk_props.clear()
        for cx in range(WORLD // CHUNK):
            for cz in range(WORLD // CHUNK):
                verts, idx = build_terrain_chunk(world, cx, cz)
                vbo = self.ctx.buffer(verts.tobytes())
                ibo = self.ctx.buffer(idx.tobytes())
                vao = self.ctx.vertex_array(
                    self.prog,
                    [(vbo, "3f 3f 3f 2f", "in_pos", "in_normal", "in_splat", "in_uv")],
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
        v9, idx = build_prop_verts(props)
        if len(v9) == 0:
            self.chunk_props[key] = (None, [])
            return
        vbo = self.ctx.buffer(v9.tobytes())
        ibo = self.ctx.buffer(idx.astype("u4").tobytes())
        vao = self.ctx.vertex_array(
            self.prop_prog, [(vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_tile")],
            index_buffer=ibo, index_element_size=4)
        self.chunk_props[key] = (vao, [vbo, ibo])

    def _basis(self, yaw: float, pitch: float) -> tuple[np.ndarray, ...]:
        fwd = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                        -math.cos(yaw) * math.cos(pitch)])
        f = fwd / np.linalg.norm(fwd)
        right = np.cross(f, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return f, right, up

    def _draw_animal(self, st: dict, res: dict, an: dict):
        prog = self.prop_prog
        run = an.get("run", 0.0)
        if "flee" in an:
            run = 1.0 if an["flee"] > 0 else 0.35
        run = min(run, 1.0)
        bm = model_rot_y(an["yaw"] + math.pi, an["x"], an["y"], an["z"])
        write_mat(prog, "u_model", bm)
        res["body"].render(mgl.TRIANGLES)
        t = st["time"] * (5.0 + 7.0 * run)
        for i, (lvao, _lv, _li) in enumerate(res["legs"]):
            phase = 0.0 if i in (0, 3) else math.pi
            swing = math.sin(t + phase) * 0.5 * run
            m = mat4_mul(bm, mat4_mul(model_translate(*res["hips"][i]), model_rot_x(swing)))
            write_mat(prog, "u_model", m)
            lvao.render(mgl.TRIANGLES)

    def _flush_flat(self, arr: np.ndarray):
        if len(arr) == 0:
            return
        self.flat_buf.orphan(arr.nbytes)
        self.flat_buf.write(arr.tobytes())
        self.flat_vao.render(mgl.TRIANGLES)

    def _billboard(self, arr: list, cx, cy, cz, right, hw, hh, y0=0.0):
        ax, az = cx - right[0] * hw, cz - right[2] * hw
        bx, bz = cx + right[0] * hw, cz + right[2] * hw
        arr.extend([
            ax, cy, az, 0.0, 0.0, bx, cy, bz, 1.0, 0.0, ax, cy + hh, az, 0.0, 1.0,
            bx, cy, bz, 1.0, 0.0, bx, cy + hh, bz, 1.0, 1.0, ax, cy + hh, az, 0.0, 1.0,
        ])
        del ax, az, bx, bz

    _HAND_TILT = {"рука": 0.35, "топор": 0.45, "копьё": 1.10, "еда": 0.05}

    def _hand_key(self, st: dict) -> tuple[str, float]:
        slot = st.get("selected", 3)
        if slot == 0 and st["tools"]["топор"]:
            return "топор", 0.5
        if slot == 1 and st["tools"]["копьё"]:
            return "копьё", 0.6
        if slot == 2:
            return "еда", 0.5
        return "рука", 0.45

    def _draw_viewmodel(self, st: dict, eye, f, right, up):
        prog = self.prop_prog
        key, cd = self._hand_key(st)
        t = st["time"]
        acd = st.get("attack_cd", 0.0)
        p = 1.0 - min(max(acd, 0.0), cd) / cd
        swing = math.sin(min(p, 1.0) * math.pi) * 1.15 if acd > 0 else 0.0
        bob = 0.02 * math.sin(t * 2.1) + 0.012 * math.sin(t * 3.7)
        tilt = self._HAND_TILT[key] + swing
        B = np.column_stack((right, up, f))
        c, s = math.cos(tilt), math.sin(tilt)
        rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        ct, stn = math.cos(0.30), math.sin(0.30)
        rz = np.array([[ct, -stn, 0], [stn, ct, 0], [0, 0, 1]])
        M = np.eye(4, dtype="f4")
        M[0:3, 0:3] = B @ (rx @ rz)
        off = np.array([0.34, -0.36 + bob - 0.10 * swing, 0.62])
        M[0:3, 3] = eye + B @ off
        self.ctx.clear(depth=1.0)
        write_mat(prog, "u_model", M)
        self.hand_res[key]["vao"].render(mgl.TRIANGLES)
        write_mat(prog, "u_model", np.eye(4))

    def draw_world(self, st: dict, sun: dict):
        ctx = self.ctx
        eye = np.array([st["px"], st["py"], st["pz"]])
        self.view = look_at(eye, st["yaw"], st["pitch"])
        pv = self.proj @ self.view
        planes = frustum_planes(pv)
        fog_c = tuple(np.clip(sun["sky"], 0, 1))

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

        prog = self.prog
        write_mat(prog, "u_pv", pv)
        write_mat(prog, "u_model", np.eye(4))
        self.world_tex.use(0)
        prog["u_atlas"].value = 0
        tf = (1.0 / ATLAS_COLS, 1.0 / ATLAS_ROWS)
        prog["u_tf"].value = tf
        prog["u_off_grass"].value = atlas_tile_uv(T_GRASS)
        prog["u_off_rock"].value = atlas_tile_uv(T_ROCK)
        prog["u_off_sand"].value = atlas_tile_uv(T_SAND)
        prog["u_sun_dir"].value = tuple(sun["sun_dir"])
        prog["u_light"].value = tuple(sun["light"])
        prog["u_amb"].value = sun["amb"]
        prog["u_cam"].value = (st["px"], st["py"], st["pz"])
        prog["u_fog_color"].value = fog_c
        prog["u_fog"].value = (FOG_NEAR, FOG_FAR)

        pprog = self.prop_prog
        write_mat(pprog, "u_pv", pv)
        pprog["u_atlas"].value = 0
        pprog["u_tf"].value = tf
        pprog["u_sun_dir"].value = tuple(sun["sun_dir"])
        pprog["u_light"].value = tuple(sun["light"])
        pprog["u_amb"].value = sun["amb"]
        pprog["u_cam"].value = (st["px"], st["py"], st["pz"])
        pprog["u_fog_color"].value = fog_c
        pprog["u_fog"].value = (FOG_NEAR, FOG_FAR)

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

        for fr in st["fires"]:
            if math.hypot(fr["x"] - st["px"], fr["z"] - st["pz"]) > FOG_FAR:
                continue
            write_mat(pprog, "u_model", model_rot_y(0.0, fr["x"], fr["y"], fr["z"]))
            self.fire_res["vao"].render(mgl.TRIANGLES)
        write_mat(pprog, "u_model", np.eye(4))

        for w in st["wolves"]:
            if math.hypot(w["x"] - st["px"], w["z"] - st["pz"]) > FOG_FAR:
                continue
            self._draw_animal(st, self.wolf_res, w)
        for b in st["boars"]:
            if math.hypot(b["x"] - st["px"], b["z"] - st["pz"]) > FOG_FAR:
                continue
            self._draw_animal(st, self.boar_res, b)
        write_mat(pprog, "u_model", np.eye(4))

        ctx.enable(mgl.BLEND)
        ctx.depth_mask = False
        fp = self.ftex_prog
        write_mat(fp, "u_pv", pv)
        self.world_tex.use(0)
        fp["u_tex"].value = 0
        fp["u_tf"].value = tf
        t = st["time"]
        fp["u_scroll"].value = (math.sin(t * 0.7) * 0.05, t * 0.35)
        fp["u_off"].value = atlas_tile_uv(T_FLAME)
        fp["u_mul"].value = (1.0, 1.0, 1.0)
        fp["u_alpha"].value = 1.0
        flame = self.build_flame(st, right)
        self._flush_flat(flame)
        lmul = tuple(np.clip(np.asarray(sun["light"]) * 0.8 + sun["amb"], 0, 1))
        if st["pickups"]:
            fp["u_scroll"].value = (0.0, 0.0)
            fp["u_off"].value = atlas_tile_uv(T_RED)
            fp["u_mul"].value = lmul
            pk_arr = []
            _f, pk_right, _u = self._basis(st["yaw"], 0.0)
            for pk in st["pickups"]:
                if math.hypot(pk["x"] - st["px"], pk["z"] - st["pz"]) > FOG_FAR:
                    continue
                bob = math.sin(t * 3.0 + pk["x"]) * 0.06
                self._billboard(pk_arr, pk["x"], pk["y"] + 0.42 + bob, pk["z"], pk_right, 0.26, 0.34)
            self._flush_flat(pk_arr)
        if self.water_res is None:
            self._build_water()
        if st["py"] - EYE_H < WATER_Y + 25.0:
            fp["u_scroll"].value = (t * 0.25, t * 0.13)
            fp["u_off"].value = atlas_tile_uv(T_WATER)
            fp["u_mul"].value = lmul
            fp["u_alpha"].value = 0.66
            self.water_res["vao"].render(mgl.TRIANGLES)
            fp["u_alpha"].value = 1.0
        ctx.depth_mask = True
        ctx.disable(mgl.BLEND)
        if st["started"] and not st["dead"]:
            self._draw_viewmodel(st, eye, f, right, up)

    def _build_water(self):
        xs = np.arange(0, WORLD + 1, 4.0)
        xx, zz = np.meshgrid(xs, xs, indexing="ij")
        m = xx.size
        data = np.empty((m, 5), dtype="f4")
        data[:, 0] = xx.ravel()
        data[:, 1] = WATER_Y
        data[:, 2] = zz.ravel()
        data[:, 3] = xx.ravel() * 0.25
        data[:, 4] = zz.ravel() * 0.25
        lix, liz = np.meshgrid(np.arange(len(xs) - 1), np.arange(len(xs) - 1), indexing="ij")
        a = (lix * len(xs) + liz).ravel()
        b, c, d = a + 1, a + len(xs), a + len(xs) + 1
        idx = np.stack([a, b, d, a, d, c], axis=1).ravel().astype("u4")
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        vao = self.ctx.vertex_array(
            self.ftex_prog, [(vbo, "3f 2f", "in_pos", "in_uv")],
            index_buffer=ibo, index_element_size=4)
        self.water_res = {"vao": vao, "bufs": (vbo, ibo)}
        return self.water_res

    def build_flame(self, st: dict, cam_right) -> np.ndarray:
        fwdh = np.array([cam_right[2], 0.0, -cam_right[0]])
        arr = []
        t = st["time"]
        for fr in st["fires"]:
            if math.hypot(fr["x"] - st["px"], fr["z"] - st["pz"]) > FOG_FAR:
                continue
            hh = 0.8 + 0.14 * math.sin(t * 11 + fr["z"])
            for axis in (cam_right, fwdh):
                arr.extend(self._quad(fr["x"], fr["y"] + 0.08, fr["z"], axis, 0.38, hh))
        return np.asarray(arr, dtype="f4")

    def _quad(self, cx, cy, cz, axis, hw, hh):
        ax, az = cx - axis[0] * hw, cz - axis[2] * hw
        bx, bz = cx + axis[0] * hw, cz + axis[2] * hw
        return [
            ax, cy, az, 0.0, 0.0, bx, cy, bz, 1.0, 0.0, ax, cy + hh, az, 0.0, 1.0,
            bx, cy, bz, 1.0, 0.0, bx, cy + hh, bz, 1.0, 1.0, ax, cy + hh, az, 0.0, 1.0,
        ]

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

    def text_r(self, s, y, col):
        self.text(self.w - 16 - text_width(s, 2), y, s, col, 2)

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

    def draw_hotbar(self, st: dict):
        slot_w, slot_h, gap = 118, 56, 10
        total = 4 * slot_w + 3 * gap
        x0 = self.w / 2 - total / 2
        y0 = self.h - slot_h - 14
        for i, name in enumerate(HOTBAR):
            x = x0 + i * (slot_w + gap)
            sel = st["selected"] == i
            self.rect(x, y0, slot_w, slot_h, (0.08, 0.10, 0.08, 0.78 if sel else 0.52))
            border = (0.95, 0.85, 0.3, 0.95) if sel else (1, 1, 1, 0.25)
            self.rect(x, y0, slot_w, 3, border)
            self.rect(x, y0 + slot_h - 3, slot_w, 3, border)
            self.rect(x, y0, 3, slot_h, border)
            self.rect(x + slot_w - 3, y0, 3, slot_h, border)
            dim = 0.35 if (i == 0 and not st["tools"]["топор"]) or \
                         (i == 1 and not st["tools"]["копьё"]) else 1.0
            self.text(x + 12, y0 + 10, str(i + 1), (1, 0.9, 0.4, 0.9))
            self.text(x + 12, y0 + 32, name, (1, 1, 1, 0.92 * dim))

    def draw_inventory(self, st: dict):
        inv = st["inv"]
        self.rect(0, 0, self.w, self.h, (0, 0, 0, 0.55))
        pw, ph = 460, 330
        px = self.w / 2 - pw / 2
        py = self.h / 2 - ph / 2
        self.rect(px, py, pw, ph, (0.06, 0.08, 0.06, 0.88))
        self.rect(px, py, pw, 3, (0.6, 0.8, 0.4, 0.9))
        self.text(px + 24, py + 18, "ИНВЕНТАРЬ", (0.75, 1.0, 0.6, 1))
        rows = [
            (f"ДЕРЕВО: {inv['дерево']}", "дерево"),
            (f"КАМЕНЬ: {inv['камень']}", "камень"),
            (f"ЯГОДЫ: {inv['ягоды']}", "ягоды"),
            (f"СЫРОЕ МЯСО: {inv['сырое мясо']}", "сырое мясо"),
            (f"ЖАРЕНОЕ МЯСО: {inv['жареное мясо']}", "жареное мясо"),
        ]
        yy = py + 60
        for line, key in rows:
            has = inv[key] > 0
            self.text(px + 24, yy, line, (1, 1, 1, 0.92) if has else (0.55, 0.55, 0.55, 0.6))
            yy += 28
        self.text(px + 24, yy + 6, "ИНСТРУМЕНТЫ:", (0.6, 0.8, 0.4, 0.8))
        yy += 34
        for tool, has in (("ТОПОР", st["tools"]["топор"]), ("КОПЬЁ", st["tools"]["копьё"])):
            mark = "ЕСТЬ" if has else "-НЕТ"
            self.text(px + 24, yy, f"{tool}: {mark}", (0.55, 1, 0.55, 0.95) if has else (0.7, 0.7, 0.7, 0.7))
            yy += 28
        self.text(px + 24, py + ph - 30, f"TAB - ЗАКРЫТЬ   СИД МИРА: {st['seed']}", (1, 1, 1, 0.6))

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
        kx = 16
        for label, on in st.get("keys_ui", ()):
            self.text(kx, 54, label, (0.4, 1, 0.4, 0.95) if on else (1, 1, 1, 0.22))
            kx += text_width(label, 2) + 16

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
        if st["swim"] and not st["dead"]:
            self.centered(self.h - 52, "ПЛЫВИ: W + ПРОБЕЛ - ВВЕРХ, SHIFT - ВНИЗ", (0.6, 0.9, 1, 0.9))
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
        if st["inv_open"] and st["started"] and not st["dead"]:
            self.draw_inventory(st)
            return
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
        self.draw_hotbar(st)
