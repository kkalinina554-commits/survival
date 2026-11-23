"""
ЛЕС: ВЫЖИВАНИЕ v2 — 3D-игра на Python (glfw + moderngl + numpy).

Инвентарь, хотбар, крафты, топор, плавание, кабаны, волки, процедурный мир.
Установка:  pip install numpy glfw moderngl
Запуск:     python game.py            (случайный мир при каждом запуске)
            python game.py --seed 42  (повторяемый мир)
Тест логики (без окна):  python game.py --selftest
"""
import sys
import traceback

import game_core as gc
from game_render import Renderer


def _parse_seed(argv):
    if "--seed" in argv:
        try:
            return int(argv[argv.index("--seed") + 1])
        except (IndexError, ValueError):
            print("--seed N: ожидалось целое число, мир будет случайным")
    return None


def run() -> int:
    import glfw
    import moderngl as mgl

    seed = _parse_seed(sys.argv)

    if not glfw.init():
        print("glfw.init() не удался")
        return 1
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(gc.W_SIZE[0], gc.W_SIZE[1],
                                "ЛЕС: ВЫЖИВАНИЕ v2 — Python 3D", None, None)
    if not window:
        glfw.terminate()
        print("Не удалось создать окно OpenGL 3.3 (нужен дисплей/драйвер, X11/Wayland/Win/Mac)")
        return 1
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    try:
        glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
    except Exception:
        pass
    glfw.swap_interval(1)
    ctx = mgl.create_context()
    renderer = Renderer(ctx, (glfw.get_framebuffer_size(window)[0],
                              glfw.get_framebuffer_size(window)[1]))
    st = gc.new_state(seed)
    renderer.build_world_gpu(st)
    print("Мир готов (seed=%s): 256x256 м, %d чанков. WASD/мышь/SHIFT/SPACE; E — добыча;"
          " ЛКМ — удар; 1-4 — хотбар; TAB — инвентарь; C — крафт; F — еда; H — помощь; ESC — пауза."
          % (seed if seed is not None else "случайный", len(renderer.chunk_terrain)))

    keys_list = [glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D, glfw.KEY_SPACE,
                 glfw.KEY_E, glfw.KEY_F, glfw.KEY_C, glfw.KEY_H, glfw.KEY_ESCAPE,
                 glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_TAB,
                 glfw.KEY_LEFT_SHIFT,
                 glfw.KEY_KP_1, glfw.KEY_KP_2, glfw.KEY_KP_3, glfw.KEY_KP_4]
    prev = {k: False for k in keys_list}
    prev["_lmb"] = False
    craft_pairs = ((glfw.KEY_1, 0), (glfw.KEY_KP_1, 0), (glfw.KEY_2, 1),
                   (glfw.KEY_KP_2, 1), (glfw.KEY_3, 2), (glfw.KEY_KP_3, 2))
    hotbar_pairs = ((glfw.KEY_1, 0), (glfw.KEY_KP_1, 0), (glfw.KEY_2, 1),
                    (glfw.KEY_KP_2, 1), (glfw.KEY_3, 2), (glfw.KEY_KP_3, 2),
                    (glfw.KEY_4, 3), (glfw.KEY_KP_4, 3))

    last = glfw.get_time()
    fps_acc = fps_n = fps_show = 0.0
    slow_warned = False
    last_mouse = None

    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = min(0.05, now - last)
        last = now

        win_w, win_h = glfw.get_framebuffer_size(window)
        if (win_w, win_h) != (renderer.w, renderer.h):
            renderer.w, renderer.h = win_w, win_h
            renderer.proj = gc.perspective(gc.FOV, win_w / win_h, 0.1, 260.0)
            renderer.ortho = gc.ortho_px(win_w, win_h)

        glfw.poll_events()
        cur = {k: glfw.get_key(window, k) == glfw.PRESS for k in keys_list}
        edges = {k: cur[k] and not prev[k] for k in keys_list}
        lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        prev["_lmb"] = lmb

        mx, myv = glfw.get_cursor_pos(window)
        looking = st["started"] and glfw.get_input_mode(window, glfw.CURSOR) == glfw.CURSOR_DISABLED
        if looking and last_mouse is not None:
            st["yaw"] -= (mx - last_mouse[0]) * 0.0023
            st["pitch"] = float(min(1.45, max(-1.45,
                          st["pitch"] - (myv - last_mouse[1]) * 0.0023)))
        last_mouse = (mx, myv)

        active = (st["started"] and not st["dead"] and not st["craft_open"]
                  and not st["inv_open"] and not st["paused"])

        if edges[glfw.KEY_ESCAPE] and st["started"] and not st["dead"]:
            st["paused"] = not st["paused"]
            st["craft_open"] = False
            st["inv_open"] = False
            glfw.set_input_mode(window, glfw.CURSOR,
                                glfw.CURSOR_NORMAL if st["paused"] else glfw.CURSOR_DISABLED)

        if st["dead"]:
            if edges[glfw.KEY_SPACE]:
                st = gc.new_state(seed)
                renderer.build_world_gpu(st)
                prev = {k: cur[k] for k in keys_list}
                prev["_lmb"] = lmb
                last_mouse = None
        elif not st["started"]:
            if edges[glfw.KEY_SPACE]:
                st["started"] = True
                print("СТАРТ ИГРЫ. Если персонаж всё ещё стоит на месте - кликни мышкой по окну игры и жми W.")
        else:
            if edges[glfw.KEY_C] and not st["paused"] and not st["inv_open"]:
                st["craft_open"] = not st["craft_open"]
            if edges[glfw.KEY_TAB] and not st["paused"]:
                st["inv_open"] = not st["inv_open"]
                if st["inv_open"]:
                    st["craft_open"] = False
            if edges[glfw.KEY_H]:
                st["show_help"] = not st["show_help"]

        if st["craft_open"]:
            for key_code, idx in craft_pairs:
                if edges.get(key_code):
                    gc.try_craft(st, idx)
        elif (st["started"] and not st["paused"] and not st["dead"]
              and not st["inv_open"]):
            for key_code, idx in hotbar_pairs:
                if edges.get(key_code):
                    st["selected"] = idx

        if active:
            mv = (float(cur[glfw.KEY_W]) - float(cur[glfw.KEY_S]),
                  float(cur[glfw.KEY_D]) - float(cur[glfw.KEY_A]))
            sprint = cur[glfw.KEY_LEFT_SHIFT]
            jump = cur[glfw.KEY_SPACE]
            try:
                if edges[glfw.KEY_E]:
                    rebuilt = gc.do_interact(st)
                    if rebuilt is None:
                        gc.message(st, "РЯДОМ НИЧЕГО НЕТ")
                    else:
                        renderer.rebuild_props_gpu(rebuilt, st["chunks"][rebuilt]["props"])
                if lmb:
                    rebuilt = gc.do_attack(st)
                    if rebuilt is not None:
                        renderer.rebuild_props_gpu(rebuilt, st["chunks"][rebuilt]["props"])
                if edges[glfw.KEY_F]:
                    gc.eat_food(st)
            except Exception:
                traceback.print_exc()
                gc.message(st, "ОШИБКА - СМОТРИ КОНСОЛЬ")
            try:
                gc.update_player(st, dt, mv, sprint, jump)
                gc.update_survival(st, dt)
                gc.update_fires(st, dt)
                gc.update_wolves(st, dt)
                gc.update_boars(st, dt)
                gc.update_time(st, dt)
                st["attack_cd"] = max(0.0, st["attack_cd"] - dt)
            except Exception:
                traceback.print_exc()
                gc.message(st, "ОШИБКА - СМОТРИ КОНСОЛЬ")

        if not st["started"]:
            st["time"] += dt
        st["messages"] = [(t, ttl - dt) for t, ttl in st["messages"] if ttl - dt > 0]

        st["keys_ui"] = [
            ("W", cur[glfw.KEY_W]), ("A", cur[glfw.KEY_A]), ("S", cur[glfw.KEY_S]),
            ("D", cur[glfw.KEY_D]), ("E", cur[glfw.KEY_E]), ("F", cur[glfw.KEY_F]),
            ("C", cur[glfw.KEY_C]), ("TAB", cur[glfw.KEY_TAB]), ("1", cur[glfw.KEY_1]),
            ("2", cur[glfw.KEY_2]), ("3", cur[glfw.KEY_3]), ("4", cur[glfw.KEY_4]),
        ]
        try:
            sun = gc.sun_state(st["time"])
            renderer.draw_world(st, sun)
            if st["py"] - gc.EYE < gc.WATER_LEVEL - 0.3:
                renderer.rect(0, 0, renderer.w, renderer.h, (0.10, 0.30, 0.50, 0.35))
            renderer.draw_hud(st, sun, fps_show)
            if st["paused"] and not st["dead"]:
                renderer.rect(0, 0, renderer.w, renderer.h, (0, 0, 0, 0.45))
                renderer.centered(renderer.h / 2 - 16, "ПАУЗА", (1, 1, 1, 0.95), 4)
                renderer.centered(renderer.h / 2 + 40, "ESC - ПРОДОЛЖИТЬ", (1, 1, 1, 0.8))
            renderer.flush_hud()
        except Exception:
            traceback.print_exc()
            try:
                renderer.centered(renderer.h / 2, "ОШИБКА РЕНДЕРА - СМОТРИ КОНСОЛЬ", (1, 0.4, 0.3, 1))
                renderer.flush_hud()
            except Exception:
                pass

        fps_acc += dt
        fps_n += 1
        if fps_acc >= 0.5:
            fps_show = fps_n / fps_acc
            if not slow_warned and fps_show > 0 and fps_show < 15:
                slow_warned = True
                print(f"Низкий FPS ({fps_show:.0f}). Если картинка дёргается или замирала - напиши мне.")
            fps_acc, fps_n = 0.0, 0

        for k in keys_list:
            prev[k] = cur[k]
        glfw.swap_buffers(window)

    glfw.destroy_window(window)
    glfw.terminate()
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return gc.selftest()
    return run()


if __name__ == "__main__":
    sys.exit(main())
