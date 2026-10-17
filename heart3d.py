# -*- coding: utf-8 -*- # надо пайгейм
"""
Бьющееся 3D-сердце, которое выпускает светящиеся 3D-частицы.

Установка:   pip install pygame numpy
             (если pygame не ставится — обычно на самом новом Python 3.14 — поставьте его
              совместимый форк:  pip install pygame-ce numpy ; код менять не нужно)
Запуск:      python heart3d.py   (в PyCharm — правый клик по файлу → Run)

Управление:
    мышь (зажать левую кнопку) — вращать камеру
    колёсико                   — приблизить / отдалить
    стрелки  ↑ / ↓             — пульс чаще / реже
    Esc                        — выход
"""
import math
import sys

try:
    import numpy as np
    import pygame
except ImportError as err:
    sys.exit("Не найдена библиотека «%s». Установите:  pip install pygame numpy\n"
             "(если pygame не ставится — pip install pygame-ce numpy)" % err.name)

# ════════════════════════════ НАСТРОЙКИ ════════════════════════════
WIDTH, HEIGHT = 960, 680    # стартовый размер окна (его можно растягивать мышью)
FPS = 60
BPM = 72                    # пульс, ударов в минуту
HEART_POINTS = 30000        # точек на поверхности сердца (меньше — быстрее)
MAX_PARTICLES = 4000        # размер пула частиц
DUST_COUNT = 320            # фоновые «пылинки» — дают ощущение глубины
BEAT_SCALE = 0.15           # насколько сердце раздувается при ударе
SWAY = 1.1                  # амплитуда покачивания сердца влево-вправо, рад (0 — не качается)
SPIN_SPEED = 0.0            # постоянное вращение, рад/с (например 0.6 — крутится по кругу)
RENDER_SCALE = 2            # 1 — чётче, но медленнее; 2 — оптимально; 3 — для слабых ПК
BLOOM = 1.0                 # сила свечения (0 — выключить)
HEART_GAIN = 1.0            # яркость сердца
PARTICLE_GAIN = 0.9         # яркость частиц


# ═══════════════════════ ГЕОМЕТРИЯ 3D-СЕРДЦА ═══════════════════════
# Поверхность сердца задана формулой F(x, y, z) = 0 (z — вверх):
#   (x² + 9/4·y² + z² − 1)³ − x²z³ − 9/80·y²z³ = 0
def _heart_f(x, y, z):
    g = x * x + 2.25 * y * y + z * z - 1.0
    return g ** 3 - x * x * z ** 3 - 0.1125 * y * y * z ** 3


def _heart_grad(x, y, z):
    g2 = (x * x + 2.25 * y * y + z * z - 1.0) ** 2
    gx = 6.0 * x * g2 - 2.0 * x * z ** 3
    gy = 13.5 * y * g2 - 0.225 * y * z ** 3
    gz = 6.0 * z * g2 - 3.0 * z * z * (x * x + 0.1125 * y * y)
    return gx, gy, gz


def build_heart(n_points, rng):
    """Равномерно рассыпает n_points точек по поверхности сердца.

    Возвращает (точки, нормали) в экранных осях: X — вправо, Y — вверх,
    Z — к зрителю. Центр сердца в (0, 0, 0)."""
    m = 120_000
    i = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)            # спираль Фибоначчи на сфере
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    d = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(phi)], axis=1)

    # на каждом луче из центра ищем расстояние до поверхности (бисекцией)
    lo = np.zeros(m)
    hi = np.full(m, 2.0)
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        inside = _heart_f(d[:, 0] * mid, d[:, 1] * mid, d[:, 2] * mid) < 0.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    r = 0.5 * (lo + hi)
    p = d * r[:, None]

    gx, gy, gz = _heart_grad(p[:, 0], p[:, 1], p[:, 2])
    n = np.stack([gx, gy, gz], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    # вес луча = площадь поверхности на нём → получаем равномерную плотность точек
    w = r * r / np.maximum(np.abs((n * d).sum(axis=1)), 0.2)
    cum = np.cumsum(w)
    cum /= cum[-1]
    u = (np.arange(n_points) + rng.random(n_points)) / n_points
    idx = np.minimum(np.searchsorted(cum, u), m - 1)
    rng.shuffle(idx)
    p, n = p[idx], n[idx]
    p = p + n * rng.normal(0.0, 0.010, size=(n_points, 1))   # лёгкая «шершавость»

    # оси формулы → экранные оси (z вверх → Y, y → Z), центрируем по высоте
    pts = np.stack([p[:, 0], p[:, 2] - 0.1265, p[:, 1]], axis=1)
    nrm = np.stack([n[:, 0], n[:, 2], n[:, 1]], axis=1)
    return pts.astype(np.float32), nrm.astype(np.float32)


# ═══════════════════════════ МАТЕМАТИКА ════════════════════════════
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def block_mean(a, k):
    """Усреднение блоков k×k у массива (3, H, W); k — степень двойки, H и W кратны k."""
    while k > 1:
        a = (a[:, 0::2, 0::2] + a[:, 1::2, 0::2] + a[:, 0::2, 1::2] + a[:, 1::2, 1::2]) * 0.25
        k //= 2
    return a


def blur3(a):
    """Лёгкое размытие [1 2 1] по обеим осям (для массива (3, H, W))."""
    a = a.copy()
    a[:, :, 1:-1] = 0.25 * a[:, :, :-2] + 0.5 * a[:, :, 1:-1] + 0.25 * a[:, :, 2:]
    a[:, 1:-1, :] = 0.25 * a[:, :-2, :] + 0.5 * a[:, 1:-1, :] + 0.25 * a[:, 2:, :]
    return a


_CACHE = {}


def _buffers(shape):
    """Заготовки для tone_to_bytes: шум для дизеринга + рабочие массивы (создаются один раз)."""
    if shape not in _CACHE:
        c, h, w = shape
        noise = np.random.default_rng(1).random(shape, dtype=np.float32)
        _CACHE[shape] = (noise, np.empty(shape, np.float32), np.empty(shape, np.float32),
                         np.empty((h, w, 3), np.uint8))
    return _CACHE[shape]


def tone_to_bytes(hdr, gain=1.0, tint=None):
    """HDR (3, H, W) → байты RGB (H, W, 3) со «мягким» сжатием ярких мест."""
    noise, x, d, out8 = _buffers(hdr.shape)
    np.multiply(hdr, gain, out=x)
    if tint is not None:
        x *= tint
    np.add(x, 1.0, out=d)
    x /= d                                 # Рейнхард x/(1+x): яркое не «выжигается» в белое
    x *= 255.0
    x += noise                             # дизеринг — убирает «полосы» в тёмных градиентах
    out8[...] = x.transpose(1, 2, 0)
    return out8.tobytes()


# ═════════════════ ОТРИСОВКА ТОЧЕК В НАКОПИТЕЛЬНЫЙ БУФЕР ═════════════════
def splat_bilinear(buf, px, py, rgb):
    """Добавляет цветные точки в буфер (3, H, W) билинейно, с субпиксельной точностью."""
    _, hb, wb = buf.shape
    ok = (px >= 0) & (px < wb - 1) & (py >= 0) & (py < hb - 1)
    if not ok.all():
        px, py, rgb = px[ok], py[ok], rgb[ok]
    if px.size == 0:
        return
    fx0 = np.floor(px)
    fy0 = np.floor(py)
    fx = px - fx0
    fy = py - fy0
    base = fy0.astype(np.int32) * wb + fx0.astype(np.int32)
    idx = np.concatenate([base, base + 1, base + wb, base + wb + 1])
    wgt = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    for c in range(3):
        col = np.tile(rgb[:, c], 4)
        buf[c].reshape(-1)[:] += np.bincount(idx, weights=wgt * col, minlength=hb * wb)


def splat_soft(buf, px, py, rgb, sigma, radius=2):
    """Добавляет мягкие круглые «светлячки» (гауссово пятно (2r+1)×(2r+1))."""
    _, hb, wb = buf.shape
    cxf = np.rint(px)
    cyf = np.rint(py)
    cx = cxf.astype(np.int32)
    cy = cyf.astype(np.int32)
    ok = (cx >= radius) & (cx < wb - radius) & (cy >= radius) & (cy < hb - radius)
    if not ok.all():
        px, py, cxf, cyf, cx, cy, rgb, sigma = (v[ok] for v in (px, py, cxf, cyf, cx, cy, rgb, sigma))
    n = px.size
    if n == 0:
        return
    offs = np.arange(-radius, radius + 1, dtype=np.int32)
    offs_f = offs.astype(np.float32)
    ex = offs_f[None, :, None] + (cxf - px)[:, None, None]       # расстояния до центра пятна
    ey = offs_f[None, None, :] + (cyf - py)[:, None, None]
    w = np.exp(-(ex * ex + ey * ey) / (2.0 * sigma[:, None, None] ** 2)).reshape(n, -1)
    idx = ((cy[:, None, None] + offs[None, None, :]) * wb
           + (cx[:, None, None] + offs[None, :, None])).reshape(-1)
    for c in range(3):
        buf[c].reshape(-1)[:] += np.bincount(idx, weights=(w * rgb[:, c, None]).reshape(-1),
                                             minlength=hb * wb)


# ═══════════════════════════ ПУЛЬС «ТУК-ТУК» ═══════════════════════════
class Pulse:
    """Двойной удар сердца: огибающая силы удара и моменты ударов."""
    ATT1, ATT2, AMP2 = 0.075, 0.065, 0.55     # быстрота нарастания и сила второго удара

    def __init__(self, bpm):
        self.bpm = float(bpm)
        self.phase = 0.90                      # 0..1, 0 — момент первого удара

    @property
    def period(self):
        return 60.0 / self.bpm

    @property
    def dub_time(self):
        return min(0.30, 0.34 * self.period)  # когда после первого удара идёт второй

    def advance(self, dt):
        """Двигает время, возвращает список наступивших ударов: 'lub' / 'dub'."""
        T = self.period
        dub = self.dub_time / T
        p0 = self.phase
        p1 = p0 + dt / T
        events = []
        if p0 < dub <= p1:
            events.append("dub")
        if p1 >= 1.0:
            p1 -= 1.0
            events.append("lub")
            if p1 >= dub:
                events.append("dub")
        self.phase = p1
        return events

    @staticmethod
    def _alpha(x, a):
        x = np.maximum(x, 0.0)
        return (x / a) * np.exp(1.0 - x / a)   # быстрый рост, плавный спад

    def envelope(self, tau):
        """Сила пульса (0…1+) через tau секунд после последнего первого удара."""
        T, d = self.period, self.dub_time
        a1, a2, k2 = self.ATT1, self.ATT2, self.AMP2
        return (self._alpha(tau, a1) + k2 * self._alpha(tau - d, a2)
                + self._alpha(tau + T, a1) + k2 * self._alpha(tau + T - d, a2))


# ═════════════════════════════ ЧАСТИЦЫ ═════════════════════════════
class Particles:
    """Пул 3D-частиц: положение, скорость, время жизни, размер, цвет."""

    def __init__(self, n, rng):
        self.n = n
        self.rng = rng
        self.pos = np.zeros((n, 3), np.float32)
        self.vel = np.zeros((n, 3), np.float32)
        self.life = np.zeros(n, np.float32)          # осталось жить (≤0 — свободна)
        self.life0 = np.ones(n, np.float32)
        self.size = np.ones(n, np.float32)
        self.hue = np.zeros(n, np.float32)
        self.drag = np.ones(n, np.float32)
        self.phase = rng.random(n).astype(np.float32) * 6.28
        self.next = 0

    def spawn(self, pos, vel, life, size, hue, drag):
        k = len(pos)
        if k == 0:
            return
        idx = (self.next + np.arange(k)) % self.n
        self.next = int((self.next + k) % self.n)
        self.pos[idx] = pos
        self.vel[idx] = vel
        self.life[idx] = life
        self.life0[idx] = life
        self.size[idx] = size
        self.hue[idx] = hue
        self.drag[idx] = drag

    def update(self, dt, t):
        alive = self.life > 0
        self.life -= dt
        self.vel *= np.exp(-self.drag * dt)[:, None]
        # лёгкое закручивание вокруг вертикальной оси и «тёплый» подъём вверх
        px, py, pz = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        self.vel[:, 0] += (-pz * 0.55) * dt
        self.vel[:, 2] += (px * 0.55) * dt
        self.vel[:, 1] += 0.22 * dt
        # плавная турбулентность — частицы летят не по прямой
        self.vel[:, 0] += 0.9 * np.sin(py * 2.3 + t * 1.3 + self.phase) * dt
        self.vel[:, 1] += 0.5 * np.sin(pz * 2.1 + t * 1.7 + self.phase * 2) * dt
        self.vel[:, 2] += 0.9 * np.sin(px * 2.2 + t * 1.1 + self.phase * 3) * dt
        self.pos += self.vel * dt * alive[:, None]


# ═════════════════════════════ СЦЕНА ═════════════════════════════
BLOOM_TINT = np.array([1.0, 0.55, 0.50], np.float32).reshape(3, 1, 1)   # свечение — красное


class Scene:
    SHADOW = np.array([0.28, 0.000, 0.020], np.float32)   # цвет в тени
    BASE = np.array([1.00, 0.020, 0.045], np.float32)     # основной алый цвет
    FILL = np.array([0.55, 0.000, 0.090], np.float32)     # холодноватая подсветка справа-снизу
    SPEC = np.array([1.00, 0.58, 0.58], np.float32)       # блики
    RIM = np.array([1.00, 0.09, 0.12], np.float32)        # контурная подсветка

    def __init__(self, w, h):
        self.rng = np.random.default_rng(12345)
        rng = self.rng
        self.P0, self.N0 = build_heart(HEART_POINTS, rng)
        n = HEART_POINTS

        # волна удара бежит по сердцу снизу вверх: у каждой точки своя задержка
        yy = (self.P0[:, 1] - self.P0[:, 1].min()) / (np.ptp(self.P0[:, 1]) + 1e-6)
        self.delays = np.linspace(0.0, 0.085, 16).astype(np.float32)
        self.bin_idx = np.clip((yy * 15.0 + 0.5).astype(np.int32), 0, 15)
        self.sparkle = (0.55 + 0.9 * rng.random(n) ** 1.5).astype(np.float32)
        self.twk_phase = (rng.random(n) * 6.28).astype(np.float32)

        # фоновая пыль (живёт в объёме вокруг сердца)
        self.dust = (rng.random((DUST_COUNT, 3)) * 2 - 1) * np.array([7.0, 4.5, 6.0])
        self.dust = self.dust.astype(np.float32)
        self.dust_phase = (rng.random(DUST_COUNT) * 6.28).astype(np.float32)
        self.dust_drift = ((rng.random((DUST_COUNT, 3)) - 0.5) * 0.12).astype(np.float32)

        self.parts = Particles(MAX_PARTICLES, rng)
        self.pulse = Pulse(BPM)
        self.t = 0.0
        self.spin = 0.0
        self.cam_yaw = 0.0           # поворот камеры мышью
        self.cam_pitch = 0.0
        self.dist = 5.4              # расстояние до камеры
        self.env = 0.0
        self.R_view = rot_x(0.16)
        self.dt = 1.0 / 60.0
        self.scale_pt = np.ones(n, np.float32)
        self.R_spin = rot_y(0.0)
        self.spark_acc = 0.0
        self.light = normalize([-0.55, 0.62, 0.56])         # основной свет: сверху слева
        self.fill = normalize([0.75, -0.45, 0.35])           # заполняющий: справа снизу
        self.half = normalize(self.light + np.array([0, 0, 1], np.float32))
        self.resize(w, h)

    # ───────────── размеры / буферы ─────────────
    def resize(self, w, h):
        self.w, self.h = w, h
        rs = RENDER_SCALE
        while (w // rs) * (h // rs) > 560_000:       # на огромных окнах считаем в меньшем разрешении
            rs += 1
        self.wb = max(64, (w // rs) // 16 * 16)
        self.hb = max(64, (h // rs) // 16 * 16)
        self.kx = self.wb / w
        self.ky = self.hb / h
        self.cx = w * 0.5
        self.cy = h * 0.52
        self.f = 1.25 * min(h, 0.85 * w)
        self.acc_front = np.zeros((3, self.hb, self.wb), np.float32)
        self.acc_trail = np.zeros((3, self.hb, self.wb), np.float32)

        # красное свечение-«аура» позади сердца
        ax = (np.arange(self.wb, dtype=np.float32) - self.cx * self.kx) ** 2
        ay = (np.arange(self.hb, dtype=np.float32) - self.cy * self.ky) ** 2
        d2 = (ay[:, None] + ax[None, :]) / (0.5 * min(self.wb, self.hb)) ** 2
        self.aura = (0.80 * np.exp(-d2 / 0.30) + 0.35 * np.exp(-d2 / 1.4)).astype(np.float32)

        # тёмный фон с мягким виньетированием (+ шум против полос)
        bx = (np.arange(w, dtype=np.float32) - self.cx) ** 2
        by = (np.arange(h, dtype=np.float32) - self.cy) ** 2
        k = np.exp(-1.6 * (by[:, None] + bx[None, :]) / (0.62 * max(w, h)) ** 2)
        noise = np.random.default_rng(2).random((h, w), dtype=np.float32)
        bg = np.empty((h, w, 3), np.uint8)
        bg[..., 0] = 5 + 34 * k + noise
        bg[..., 1] = 3 * k + noise
        bg[..., 2] = 3 + 13 * k + noise
        self.bg = pygame.image.frombuffer(bg.tobytes(), (w, h), "RGB").convert()

    # ───────────── обновление ─────────────
    def update(self, dt, user_dx=0.0, user_dy=0.0):
        self.dt = dt
        self.t += dt
        t = self.t
        self.spin = SPIN_SPEED * t + SWAY * math.sin(0.5 * t)
        self.cam_yaw += user_dx
        self.cam_pitch = float(np.clip(self.cam_pitch + user_dy, -1.3, 1.3))

        events = self.pulse.advance(dt)
        tau = self.pulse.phase * self.pulse.period
        self.env = float(self.pulse.envelope(tau))
        table = self.pulse.envelope(tau - self.delays).astype(np.float32)
        self.scale_pt = 1.0 + BEAT_SCALE * table[self.bin_idx]
        self.R_spin = rot_y(self.spin)

        # матрица вида: небольшое «дыхание» камеры + то, что задал пользователь мышью
        pitch = 0.16 + 0.06 * math.sin(t * 0.37) + self.cam_pitch
        self.R_view = rot_x(pitch) @ rot_y(self.cam_yaw)

        for ev in events:
            if ev == "lub":
                self.burst(110, 1.0)
                self.ring(64)
            else:
                self.burst(60, 0.75)
        # постоянный «звёздный дождь» искр
        self.spark_acc += 55.0 * dt
        k = int(self.spark_acc)
        if k:
            self.spark_acc -= k
            self.burst(k, 0.45)

        self.parts.update(dt, t)

    def burst(self, count, strength):
        """Выпускает count частиц с поверхности сердца наружу."""
        rng = self.rng
        idx = rng.integers(0, len(self.P0), count)
        pos = (self.P0[idx] * self.scale_pt[idx, None]) @ self.R_spin.T
        nrm = self.N0[idx] @ self.R_spin.T
        direction = nrm + rng.normal(0.0, 0.38, (count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
        speed = strength * (0.55 + 2.0 * rng.random(count) ** 1.4)
        life = 1.5 + 2.2 * rng.random(count)
        size = 0.010 + 0.028 * rng.random(count) ** 2
        bokeh = rng.random(count) < 0.05                 # ~5 % частиц — крупные «боке»-шары
        size = np.where(bokeh, 0.07 + 0.05 * rng.random(count), size)
        life = np.where(bokeh, life + 1.5, life)
        self.parts.spawn(pos * 1.03, direction * speed[:, None], life,
                         size,
                         rng.random(count), 0.45 + 0.6 * rng.random(count))

    def ring(self, count):
        """Кольцо-«ударная волна», расходящееся от сердца при каждом ударе."""
        rng = self.rng
        a = rng.random(count) * 6.2832
        c, s = np.cos(a), np.sin(a)
        pos = np.stack([c * 1.25, np.full(count, -0.05), s * 1.25], axis=1)
        sp = 2.4 + 0.25 * rng.random(count)
        vel = np.stack([c * sp, np.zeros(count), s * sp], axis=1)
        self.parts.spawn(pos, vel, 0.9 + 0.2 * rng.random(count),
                         np.full(count, 0.014), np.full(count, 0.12), np.full(count, 2.3))

    # ───────────── отрисовка ─────────────
    def draw(self, screen):
        front, trail = self.acc_front, self.acc_trail
        front.fill(0.0)
        trail *= math.exp(-self.dt / 0.045)            # хвосты частиц быстро гаснут

        self.draw_heart(front)
        self.draw_dust(front)
        self.draw_particles(trail)

        hdr = front + trail
        a = (0.05 + 0.20 * self.env) * self.aura
        hdr[0] += a * 0.55
        hdr[1] += a * 0.020
        hdr[2] += a * 0.060

        # основной слой + свечение (bloom) двух радиусов — всё складывается поверх тёмного фона
        screen.blit(self.bg, (0, 0))
        screen.blit(self.layer(hdr), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if BLOOM > 0:
            b1 = block_mean(hdr, 4)
            b2 = block_mean(b1, 4)
            for glow, gain in ((blur3(b1), 0.60 * BLOOM), (blur3(blur3(b2)), 1.00 * BLOOM)):
                screen.blit(self.layer(glow, gain, BLOOM_TINT), (0, 0),
                            special_flags=pygame.BLEND_RGB_ADD)

    def layer(self, hdr, gain=1.0, tint=None):
        """HDR-массив → поверхность размером с окно (растягиваем гладко)."""
        hb, wb = hdr.shape[1:]
        raw = tone_to_bytes(hdr, gain, tint)          # raw держим в переменной, пока жива поверхность
        small = pygame.image.frombuffer(raw, (wb, hb), "RGB").convert()
        return pygame.transform.smoothscale(small, (self.w, self.h))

    def draw_heart(self, buf):
        t = self.t
        M = self.R_view @ self.R_spin
        P = self.P0 * self.scale_pt[:, None]
        Pv = P @ M.T
        Nv = self.N0 @ M.T
        k = self.f / (self.dist - Pv[:, 2])
        sx = (self.cx + Pv[:, 0] * k) * self.kx
        sy = (self.cy - Pv[:, 1] * k) * self.ky

        nz = Nv[:, 2]                                   # «смотрит ли» точка на камеру
        ndl = Nv @ self.light
        diff = np.clip((ndl + 0.30) / 1.30, 0.0, 1.0)   # мягкий «обёрнутый» диффуз
        diff = diff * diff
        fill = np.clip(Nv @ self.fill, 0.0, 1.0)
        s = np.clip(Nv @ self.half, 0.0, 1.0)
        s2 = s * s
        s4 = s2 * s2
        s8 = s4 * s4
        spec = s8 * s8 * s8 * s4 * s4                   # ≈ s^40 — острый глянцевый блик
        sheen = s4 * 0.22                               # и широкий мягкий отсвет
        rim = 1.0 - np.clip(nz, 0.0, 1.0)
        rim = rim * rim * rim
        vis = 0.05 + 0.95 * np.clip((nz + 0.10) / 0.28, 0.0, 1.0)   # задняя сторона — еле видна

        rgb = (self.SHADOW + diff[:, None] * (self.BASE - self.SHADOW)
               + (fill * 0.55)[:, None] * self.FILL
               + (spec * 1.6 + sheen)[:, None] * self.SPEC
               + (rim * 1.10)[:, None] * self.RIM)
        twinkle = 0.84 + 0.16 * np.sin(t * 6.0 + self.twk_phase)
        gain = vis * self.sparkle * twinkle * (0.62 + 0.80 * self.env) * HEART_GAIN
        rgb *= gain[:, None]
        splat_bilinear(buf, sx, sy, rgb.astype(np.float32))

    def draw_dust(self, buf):
        t = self.t
        box = np.array([14.0, 9.0, 12.0], np.float32)
        self.dust = (self.dust + self.dust_drift * self.dt + box / 2) % box - box / 2
        pv = self.dust @ self.R_view.T
        depth = self.dist - pv[:, 2]
        ok = depth > 1.0
        k = self.f / np.maximum(depth, 1e-3)
        sx = (self.cx + pv[:, 0] * k) * self.kx
        sy = (self.cy - pv[:, 1] * k) * self.ky
        tw = 0.5 + 0.5 * np.sin(t * 1.7 + self.dust_phase)
        inten = (0.05 + 0.20 * tw) * np.clip(1.4 - depth * 0.09, 0.2, 1.0) * ok
        rgb = inten[:, None] * np.array([1.0, 0.25, 0.35], np.float32)
        splat_bilinear(buf, sx, sy, rgb.astype(np.float32))

    def draw_particles(self, buf):
        p = self.parts
        sel = np.nonzero(p.life > 0)[0]
        if sel.size == 0:
            return
        pos = p.pos[sel]
        pv = pos @ self.R_view.T
        depth = self.dist - pv[:, 2]
        near = depth > 0.9
        sel, pv, depth = sel[near], pv[near], depth[near]
        if sel.size == 0:
            return
        k = self.f / depth
        sx = (self.cx + pv[:, 0] * k) * self.kx
        sy = (self.cy - pv[:, 1] * k) * self.ky

        age = 1.0 - p.life[sel] / p.life0[sel]
        fade = (1.0 - age) ** 1.3 * np.clip(age * 14.0, 0.0, 1.0)
        tw = 0.75 + 0.25 * np.sin(self.t * 11.0 + p.phase[sel])
        keys = (0.0, 0.12, 0.45, 1.0)
        r = np.interp(age, keys, (1.00, 1.00, 0.95, 0.40))
        g = np.interp(age, keys, (0.85, 0.30, 0.07, 0.00))
        b = np.interp(age, keys, (0.90, 0.42, 0.13, 0.02)) * (0.6 + 0.8 * p.hue[sel])
        rad = p.size[sel] * k * self.kx                    # радиус в пикселях буфера
        bokeh = p.size[sel] > 0.05
        inten = fade * tw * PARTICLE_GAIN * np.where(
            bokeh, 0.30, 1.0 + 0.55 * np.clip(rad, 0.0, 3.0))
        rgb = (np.stack([r, g, b], axis=1) * inten[:, None]).astype(np.float32)
        sx = sx.astype(np.float32)
        sy = sy.astype(np.float32)
        small = ~bokeh
        sigma = np.clip(0.50 + 0.42 * rad, 0.6, 1.25).astype(np.float32)
        splat_soft(buf, sx[small], sy[small], rgb[small], sigma[small], radius=3)
        if bokeh.any():
            sigma_b = np.clip(0.55 * rad, 1.5, 4.0).astype(np.float32)
            splat_soft(buf, sx[bokeh], sy[bokeh], rgb[bokeh], sigma_b[bokeh], radius=8)


# ═════════════════════════════ ЗАПУСК ═════════════════════════════
TITLE = "Бьющееся 3D-сердце   |   мышь — вращение,  колёсико — зум,  ↑/↓ — пульс,  Esc — выход"


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    scene = Scene(*screen.get_size())

    dragging = False
    running = True
    fps_timer = 0.0
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        dx = dy = 0.0
        fps_timer += dt
        if fps_timer >= 1.0:                     # раз в секунду показываем FPS в заголовке
            fps_timer = 0.0
            pygame.display.set_caption("%s   |   %.0f FPS" % (TITLE, clock.get_fps()))
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_UP:
                    scene.pulse.bpm = min(180.0, scene.pulse.bpm + 6)
                elif e.key == pygame.K_DOWN:
                    scene.pulse.bpm = max(36.0, scene.pulse.bpm - 6)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                dx += e.rel[0] * 0.007
                dy += e.rel[1] * 0.007
            elif e.type == pygame.MOUSEWHEEL:
                scene.dist = float(np.clip(scene.dist * (0.92 ** e.y), 3.0, 14.0))

        screen = pygame.display.get_surface()
        w, h = screen.get_size()
        if w < 96 or h < 96:                     # окно свёрнуто
            continue
        if (w, h) != (scene.w, scene.h):
            scene.resize(w, h)

        scene.update(dt, dx, dy)
        scene.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
