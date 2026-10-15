"""NEON ARENA 3D - Panda3D, pandas y machine learning.

Controles:
    W: mantener vuelo; flechas: dirigir el vuelo en el plano
    Espacio: dash
    M: activar o silenciar audio
    R: reiniciar
    Esc: salir

Instalacion:
    python -m pip install -r requirements.txt

Ejecucion:
    python neon_arena_3d.py
"""

import math
import random
import struct
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise SystemExit(
        'Falta una dependencia. Ejecuta: python -m pip install -r requirements.txt'
    ) from exc

try:
    from direct.gui.OnscreenText import OnscreenText
    from direct.gui.DirectGui import (
        DirectButton,
        DirectFrame,
        DirectLabel,
        DirectWaitBar,
    )
    from direct.showbase.ShowBase import ShowBase
    from direct.showbase.ShowBaseGlobal import globalClock
    from direct.task import Task
    from panda3d.core import (
        AmbientLight,
        AudioSound,
        CardMaker,
        DirectionalLight,
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        Filename,
        Material,
        TextNode,
        TransparencyAttrib,
        UserDataAudio,
        Vec3,
        Vec4,
        WindowProperties,
    )
except ImportError as exc:
    raise SystemExit('Falta Panda3D. Ejecuta: python -m pip install panda3d') from exc


ARENA = 23
POINTS_PER_LEVEL = 50
CSV_PATH = Path(__file__).with_name('estadisticas_neon_arena.csv')
AUDIO_RATE = 22_050

PATROL = 0
CHASE = 1
INTERCEPT = 2
FLANK = 3
GUARD = 4
RETREAT = 5

BEHAVIOR_NAMES = {
    PATROL: 'PATRULLA',
    CHASE: 'PERSECUCION',
    INTERCEPT: 'INTERCEPCION',
    FLANK: 'FLANQUEO',
    GUARD: 'DEFENSA',
    RETREAT: 'RETIRADA',
}

AI_FEATURES = [
    'distancia',
    'velocidad_cierre',
    'velocidad_jugador',
    'vida_jugador',
    'dash_disponible',
    'resistencia_dron',
    'distancia_pared',
    'distancia_energia',
    'aliados_cercanos',
    'nivel',
]


def cube(parent, pos, scale, rgba):
    """Crea un cubo procedural sin depender de modelos externos."""
    vertices = [
        (-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
        (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    fmt = GeomVertexFormat.getV3()
    data = GeomVertexData('cube', fmt, Geom.UHStatic)
    writer = GeomVertexWriter(data, 'vertex')
    for x, y, z in vertices:
        writer.addData3(x, y, z)
    triangles = GeomTriangles(Geom.UHStatic)
    for a, b, c, d in faces:
        triangles.addVertices(a, b, c)
        triangles.addVertices(a, c, d)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode('cube')
    node.addGeom(geom)
    item = parent.attachNewNode(node)
    item.setPos(*pos)
    item.setScale(*scale)
    item.setColor(*rgba)
    item.setTwoSided(True)
    material = Material('neon_cube_material')
    material.setDiffuse(Vec4(*rgba))
    material.setEmission(Vec4(rgba[0] * .22, rgba[1] * .22, rgba[2] * .22, rgba[3]))
    item.setMaterial(material, 1)
    return item


def energy_sphere(parent, pos, radius, rgba, name='energy_sphere'):
    """Esfera UV de baja poligonizacion, construida enteramente en memoria."""
    rings, sides = 8, 12
    data = GeomVertexData(name, GeomVertexFormat.getV3(), Geom.UHStatic)
    writer = GeomVertexWriter(data, 'vertex')
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for side in range(sides):
            theta = math.tau * side / sides
            writer.addData3(
                radius * math.sin(phi) * math.cos(theta),
                radius * math.sin(phi) * math.sin(theta),
                radius * math.cos(phi),
            )
    tris = GeomTriangles(Geom.UHStatic)
    for ring in range(rings):
        for side in range(sides):
            a = ring * sides + side
            b = ring * sides + (side + 1) % sides
            c = (ring + 1) * sides + (side + 1) % sides
            d = (ring + 1) * sides + side
            tris.addVertices(a, b, c)
            tris.addVertices(a, c, d)
    geom = Geom(data)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    sphere = parent.attachNewNode(node)
    sphere.setPos(*pos)
    sphere.setColor(*rgba)
    sphere.setTwoSided(True)
    sphere.setLightOff(1)
    if rgba[3] < 1:
        sphere.setTransparency(TransparencyAttrib.MAlpha)
    return sphere


def faceted_mesh(parent, name, vertices, triangles, rgba):
    """Malla triangular con normales planas para piezas low-poly 3D."""
    data = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vertex_writer = GeomVertexWriter(data, 'vertex')
    normal_writer = GeomVertexWriter(data, 'normal')
    primitive = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        a, b, c = (Vec3(*vertices[index]) for index in triangle)
        normal = (b - a).cross(c - a)
        if normal.lengthSquared() < 1e-8:
            normal = Vec3(0, 0, 1)
        else:
            normal.normalize()
        start = data.getNumRows()
        for point in (a, b, c):
            vertex_writer.addData3(point)
            normal_writer.addData3(normal)
        primitive.addVertices(start, start + 1, start + 2)
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    piece = parent.attachNewNode(node)
    piece.setColor(*rgba)
    piece.setTwoSided(True)
    material = Material(f'{name}_material')
    material.setDiffuse(Vec4(*rgba))
    material.setSpecular(Vec4(.92, .96, 1.0, 1))
    material.setShininess(72)
    material.setAmbient(Vec4(rgba[0] * .42, rgba[1] * .42, rgba[2] * .42, rgba[3]))
    material.setEmission(Vec4(rgba[0] * .16, rgba[1] * .16, rgba[2] * .16, rgba[3]))
    piece.setMaterial(material, 1)
    return piece


def apply_metal_finish(node, rgba, shine=64):
    """Aplica reflejo especular y base metálica a una pieza del dron."""
    material = Material(f'{node.getName()}_metal')
    material.setDiffuse(Vec4(*rgba))
    material.setAmbient(Vec4(rgba[0] * .38, rgba[1] * .38, rgba[2] * .38, rgba[3]))
    material.setSpecular(Vec4(.95, .98, 1.0, 1))
    material.setShininess(shine)
    material.setEmission(Vec4(rgba[0] * .09, rgba[1] * .09, rgba[2] * .09, rgba[3]))
    node.setMaterial(material, 1)
    node.clearLight()
    return node


def normalized(vector, fallback=(0, 0, 0)):
    """Devuelve un Vec3 normalizado sin dividir entre cero."""
    result = Vec3(vector)
    if result.lengthSquared() > 1e-8:
        result.normalize()
        return result
    return Vec3(*fallback)


def synth_pcm(segments, waveform='sine', seed=0):
    """Genera audio mono PCM de 16 bits completamente en memoria.

    Cada segmento es (frecuencia_inicial, frecuencia_final, duracion, volumen).
    """
    frames = bytearray()
    phase = 0.0
    rng = random.Random(seed)

    for start_hz, end_hz, duration, volume in segments:
        sample_count = max(1, int(AUDIO_RATE * duration))
        attack = max(1, min(int(AUDIO_RATE * .012), sample_count // 3))
        release = max(1, min(int(AUDIO_RATE * .035), sample_count // 3))

        for index in range(sample_count):
            progress = index / max(1, sample_count - 1)
            frequency = start_hz + (end_hz - start_hz) * progress
            phase += math.tau * frequency / AUDIO_RATE
            envelope = min(
                1.0,
                (index + 1) / attack,
                (sample_count - index) / release,
            )
            tone = math.sin(phase) + .18 * math.sin(phase * 2.0)
            if waveform == 'noise':
                tone = .72 * rng.uniform(-1.0, 1.0) + .28 * math.sin(phase)
            elif waveform == 'square':
                tone = .82 if math.sin(phase) >= 0 else -.82
            elif waveform == 'brass':
                tone = (
                    math.sin(phase) + .46 * math.sin(phase * 2)
                    + .24 * math.sin(phase * 3) + .12 * math.sin(phase * 4)
                )
            sample = max(-1.0, min(1.0, tone * volume * envelope / 1.18))
            frames.extend(struct.pack('<h', int(sample * 32_767)))

    return bytes(frames)


def synth_drone_hum(duration=2.0):
    """Zumbido continuo y suave con vibrato, parecido al vuelo de un insecto."""
    frames = bytearray()
    phase = 0.0
    sample_count = int(AUDIO_RATE * duration)
    for index in range(sample_count):
        t = index / AUDIO_RATE
        frequency = 180 + 12 * math.sin(math.tau * 3 * t)
        phase += math.tau * frequency / AUDIO_RATE
        tremolo = .70 + .12 * math.sin(math.tau * 4.5 * t)
        tone = (
            .58 * math.sin(phase)
            + .25 * math.sin(phase * 2)
            + .12 * math.sin(phase * 3)
            + .05 * math.sin(phase * 5)
        ) * tremolo
        frames.extend(struct.pack('<h', int(max(-1, min(1, tone * .72)) * 32767)))
    return bytes(frames)


class NeonArena(ShowBase):
    """Juego principal."""

    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(.015, .025, .065, 1)

        props = WindowProperties()
        props.setTitle('NEON ARENA 3D | MACHINE LEARNING BATTLE')
        props.setSize(1280, 720)
        if self.win is not None and hasattr(self.win, 'requestProperties'):
            self.win.requestProperties(props)

        self.rng = random.Random()
        self.audio_enabled = True
        self.audio_available = False
        self.audio_sources = {}
        self.sounds = {}
        self.ambient = None
        self.web_shots = []
        self.web_cooldown = 0.0
        self.web_max_ammo = 6
        self.web_ammo = 4
        self.flight_height = 0.0

        physical_keys = (
            'w', 'a', 's', 'd',
            'arrow_up', 'arrow_down', 'arrow_left', 'arrow_right',
            'space',
        )
        self.keys = {key: False for key in physical_keys}
        for key in physical_keys[:-1]:
            self.accept(key, self.set_key, [key, True])
            self.accept(f'{key}-up', self.set_key, [key, False])
        self.accept('space', self.set_dash_key, [True])
        self.accept('space-up', self.set_dash_key, [False])
        self.accept('f', self.fire_web)
        self.accept('m', self.toggle_audio)
        self.accept('p', self.toggle_pause)
        self.accept('r', self.restart_game)
        self.accept('escape', self.userExit)

        self.build_ai()
        self.best_score = self.load_best_score()
        self.build_world()
        self.build_hud()
        self.build_audio()
        self.game_mode = 'PC vs 1 jugador'
        self.game_started = False
        self.paused = False
        self.build_mode_menu()
        # Fanfarria original de apertura al aparecer el menú principal.
        self.play_sound('intro')
        self.taskMgr.add(self.update_game, 'update_game')

    def build_mode_menu(self):
        """Muestra la seleccion inicial de modalidad de juego."""
        self.mode_menu = DirectFrame(
            parent=self.aspect2d,
            frameColor=(.015, .035, .095, .96),
            frameSize=(-.86, .86, -.78, .78),
            relief=1,
        )
        DirectLabel(
            parent=self.mode_menu,
            text='NEON ARENA 3D',
            scale=.105,
            pos=(0, 0, .57),
            text_fg=(.25, .91, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        DirectLabel(
            parent=self.mode_menu,
            text='ELIGE COMO QUIERES JUGAR',
            scale=.052,
            pos=(0, 0, .38),
            text_fg=(.82, .9, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        choices = (
            ('PC contra 1 usuario', 'PC vs 1 jugador'),
            ('PC contra 2 usuarios', 'PC vs 2 jugadores'),
            ('Usuario contra usuario', '1 jugador vs 1 jugador'),
        )
        self.mode_buttons = []
        for index, (label, mode) in enumerate(choices):
            button = DirectButton(
                parent=self.mode_menu,
                text=label,
                text_scale=.052,
                text_fg=(.94, .98, 1, 1),
                frameColor=(.035, .22, .34, 1),
                frameSize=(-.58, .58, -.09, .09),
                relief=1,
                pos=(0, 0, .12 - index * .23),
                command=self.select_game_mode,
                extraArgs=[mode],
                rolloverSound=None,
                clickSound=None,
            )
            self.mode_buttons.append(button)
        DirectLabel(
            parent=self.mode_menu,
            text='Selecciona una opcion para comenzar',
            scale=.037,
            pos=(0, 0, -.62),
            text_fg=(.57, .77, .9, 1),
            frameColor=(0, 0, 0, 0),
        )

    def select_game_mode(self, mode):
        """Guarda la opcion elegida e inicia la partida."""
        self.game_mode = mode
        self.game_started = True
        self.mode_menu.hide()
        self.bottom_bar.show()
        self.game_controls.show()
        self.restart_game()
        self.show_message(f'MODO: {mode.upper()}', 2.5)

    def set_key(self, key, value):
        self.keys[key] = value

    def set_dash_key(self, value):
        if value and not self.keys['space']:
            self.dash_requested = True
        self.keys['space'] = value

    @staticmethod
    def build_web_net(parent, color=(.78, .96, 1, 1)):
        """Teje una red circular 3D con hilos finos y un nudo central."""
        def thread(start, end, rgba, thickness):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            bar = cube(
                parent,
                ((start[0] + end[0]) * .5, (start[1] + end[1]) * .5, .035),
                (thickness, length, thickness), rgba,
            )
            bar.setH(math.degrees(math.atan2(dx, dy)))

        radius = .92
        for index in range(12):
            angle = math.tau * index / 12
            thread((0, 0), (radius * math.cos(angle), radius * math.sin(angle)),
                   color, .022)
        for ring_index, ring_radius in enumerate((.25, .49, .73, .92)):
            points = [
                (ring_radius * math.cos(math.tau * i / 16),
                 ring_radius * math.sin(math.tau * i / 16))
                for i in range(16)
            ]
            for index in range(16):
                thread(points[index], points[(index + 1) % 16],
                       color if ring_index % 2 == 0 else (.24, .68, .9, 1), .018)
        energy_sphere(parent, (0, 0, .05), .105, (.55, .92, 1, 1), 'web_knot')

    def fire_web(self):
        """Dispara una red en línea recta hacia delante."""
        if (not self.game_started or self.paused or self.finished
                or self.web_cooldown > 0):
            return
        if self.web_ammo <= 0:
            self.show_message('SIN TELARAÑAS: RECOGE UNA CRUZ ROJA', 1.3)
            return
        player_pos = self.player.getPos()
        direction = Vec3(self.facing_direction)
        if direction.lengthSquared() < 1e-8:
            direction = Vec3(0, 1, 0)
        direction.z = 0
        direction = normalized(direction, (0, 1, 0))
        projectile = self.scene.attachNewNode('web_projectile')
        muzzle = player_pos + direction * 1.05
        projectile.setPos(muzzle.x, muzzle.y, 1.35)
        projectile.setScale(.28)
        self.build_web_net(projectile)
        self.web_shots.append({
            'node': projectile,
            'velocity': direction * 19,
            'age': 0.0,
        })
        self.web_cooldown = 1.65
        self.web_ammo -= 1
        self.play_sound('web_shot')
        self.session_events.append({
            'segundo': round(self.elapsed, 2),
            'evento': 'disparo_telaraña',
            'nivel': self.level,
        })

    def update_web_shots(self, dt):
        self.web_cooldown = max(0.0, self.web_cooldown - dt)
        for shot in self.web_shots[:]:
            shot['age'] += dt
            node = shot['node']
            position = node.getPos() + shot['velocity'] * dt
            node.setPos(position)
            node.setScale(min(1.0, .32 + shot['age'] * 2.8))
            caught = None
            for drone in self.drones:
                if drone.get('webbed_until', 0) > self.elapsed:
                    continue
                delta = drone['node'].getPos() - position
                if delta.lengthSquared() < 1.22 ** 2:
                    caught = drone
                    break

            if caught is not None:
                node.reparentTo(caught['node'])
                node.setPos(0, 0, 1.32)
                node.setH(0)
                node.setScale(.78)
                caught['webbed_until'] = self.elapsed + .48
                caught['neutralize_at'] = self.elapsed + .48
                caught['web_node'] = node
                caught['velocity'] = Vec3()
                self.play_sound('web_catch')
                self.session_events.append({
                    'segundo': round(self.elapsed, 2),
                    'evento': 'dron_enredado',
                    'nivel': self.level,
                })
                self.show_message('TELARAÑA: ENEMIGO ATRAPADO', .55)
                self.web_shots.remove(shot)
                continue

            if (shot['age'] > 1.45 or abs(position.x) > ARENA
                    or abs(position.y) > ARENA):
                node.removeNode()
                self.web_shots.remove(shot)

    def update_tower_flames(self, dt):
        """Anima las llamas de las torres incluso detrás del menú inicial."""
        for flame in self.tower_flames:
            flame['phase'] += dt * 5.4
            pulse = math.sin(flame['phase'])
            root = flame['node']
            root.setZ(flame['base_z'] + .04 * pulse)
            root.setH(root.getH() + 28 * dt)
            root.setScale(1 + .07 * pulse, 1 + .06 * math.cos(flame['phase']),
                          .88 + .20 * (1 + pulse) * .5)

    def update_crowd(self, dt):
        """Anima suavemente los personajes del publico en las gradas."""
        for npc in self.reference_crowd:
            npc['phase'] += dt * 3.0
            wave = math.sin(npc['phase'])
            npc['node'].setZ(npc['base_z'] + .035 * wave)

    def build_audio(self):
        """Crea musica y efectos en memoria; el juego funciona si fallan."""
        sound_specs = {
            'start': (
                [(330, 440, .10, .65), (440, 660, .14, .62)], 'sine', .34,
            ),
            'intro': (
                [
                    (196, 196, .22, .62), (294, 294, .22, .58),
                    (392, 392, .24, .62), (587, 587, .58, .72),
                    (494, 494, .20, .55), (659, 659, .20, .58),
                    (784, 784, .75, .72), (587, 587, .22, .56),
                    (784, 784, .22, .62), (988, 988, .95, .72),
                ], 'brass', .40,
            ),
            'collect': (
                [(720, 980, .07, .75), (980, 1320, .10, .70)], 'sine', .38,
            ),
            'dash': ([(150, 760, .18, .62)], 'square', .25),
            'web_shot': (
                [(680, 410, .09, .45), (410, 260, .15, .52)], 'noise', .30,
            ),
            'web_catch': (
                [(520, 340, .08, .40), (340, 520, .11, .38)], 'sine', .24,
            ),
            'web_kill': (
                [(430, 690, .08, .48), (690, 980, .11, .50)], 'sine', .28,
            ),
            'hit': ([(210, 75, .26, .82)], 'noise', .42),
            'level': (
                [
                    (440, 440, .10, .66), (554, 554, .10, .66),
                    (659, 659, .10, .66), (880, 1040, .19, .72),
                ],
                'sine',
                .40,
            ),
            'game_over': (
                [
                    (420, 330, .18, .72), (330, 240, .22, .68),
                    (240, 90, .42, .62),
                ],
                'sine',
                .42,
            ),
            'crowd_cheer': ([(380, 520, .16, .8), (520, 780, .22, .9), (660, 940, .28, .85)], 'noise', .3),
            'crowd_boo': ([(240, 180, .18, .7), (190, 130, .25, .8)], 'noise', .24),
            'crowd_applause': ([(900, 1200, .035, .8), (720, 1100, .04, .75), (850, 1350, .03, .8)], 'noise', .23),
            'ambient': (
                [
                    (110, 110, .75, .42), (138, 138, .75, .38),
                    (165, 165, .75, .36), (138, 138, .75, .38),
                    (123, 123, .75, .40), (165, 165, .75, .36),
                    (185, 185, .75, .34), (138, 110, .75, .38),
                ],
                'sine',
                .11,
            ),
        }

        try:
            for index, (name, (segments, waveform, volume)) in enumerate(
                sound_specs.items()
            ):
                source = UserDataAudio(
                    rate=AUDIO_RATE,
                    channels=1,
                    remove_after_read=False,
                )
                source.append(synth_pcm(segments, waveform, seed=900 + index))
                source.done()
                sound = (
                    self.loader.loadMusic(source)
                    if name == 'ambient'
                    else self.loader.loadSfx(source)
                )
                if sound is None or sound.status() == AudioSound.BAD:
                    continue
                sound.setVolume(volume)
                self.audio_sources[name] = source
                if name == 'ambient':
                    sound.setLoop(True)
                    self.ambient = sound
                else:
                    self.sounds[name] = sound
            self.audio_available = bool(self.sounds or self.ambient)
            hum_source = UserDataAudio(
                rate=AUDIO_RATE, channels=1, remove_after_read=False,
            )
            hum_source.append(synth_drone_hum())
            hum_source.done()
            hum = self.loader.loadSfx(hum_source)
            if hum is not None and hum.status() != AudioSound.BAD:
                hum.setLoop(True)
                hum.setVolume(.16)
                self.audio_sources['drone_hum'] = hum_source
                self.sounds['drone_hum'] = hum
                self.audio_available = True
        except Exception:
            self.audio_available = False
            self.audio_sources = {}
            self.sounds = {}
            self.ambient = None

    def play_sound(self, name):
        if not self.audio_enabled:
            return
        sound = self.sounds.get(name)
        if sound is None or sound.status() == AudioSound.BAD:
            return
        sound.stop()
        try:
            sound.setTime(0)
        except (AssertionError, TypeError):
            pass
        sound.play()

    def play_ambient(self, restart=False):
        if not self.audio_enabled or self.ambient is None:
            return
        if restart:
            self.ambient.stop()
            try:
                self.ambient.setTime(0)
            except (AssertionError, TypeError):
                pass
        if self.ambient.status() != AudioSound.PLAYING:
            self.ambient.play()

    def play_drone_hum(self):
        if not self.audio_enabled or not self.game_started or self.paused or self.finished:
            return
        sound = self.sounds.get('drone_hum')
        if sound is not None and sound.status() != AudioSound.PLAYING:
            sound.play()

    def toggle_audio(self):
        if not self.audio_available:
            self.show_message('AUDIO NO DISPONIBLE EN ESTE EQUIPO', 2.0)
            return
        self.audio_enabled = not self.audio_enabled
        if self.audio_enabled:
            self.play_ambient()
            self.play_drone_hum()
            self.show_message('SONIDO ACTIVADO', 1.5)
        else:
            for sound in self.sounds.values():
                sound.stop()
            if self.ambient is not None:
                self.ambient.stop()
            self.show_message('SONIDO SILENCIADO', 1.5)

    def build_ai(self):
        """Entrena IA tactica y prepara aprendizaje del jugador en sesion."""
        training_data = self.generate_training_data(5_000)
        frame = pd.DataFrame(training_data)
        features = frame[AI_FEATURES]
        labels = frame['accion']
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=.22,
            random_state=42,
            stratify=labels,
        )
        self.ai = RandomForestClassifier(
            n_estimators=72,
            max_depth=8,
            min_samples_leaf=4,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=1,
        )
        self.ai.fit(x_train, y_train)
        self.ai_accuracy = float(accuracy_score(y_test, self.ai.predict(x_test)))
        self.ai_training_rows = len(frame)

        self.player_model = SGDClassifier(
            loss='log_loss',
            alpha=.001,
            random_state=42,
        )
        self.player_samples = 0
        self.previous_player_features = None

    @staticmethod
    def generate_training_data(sample_count):
        """Genera estados tacticos reproducibles para aprendizaje supervisado."""
        rng = random.Random(42)
        rows = []
        for _ in range(sample_count):
            distance = rng.uniform(.5, 34)
            closing_speed = rng.uniform(-12, 12)
            player_speed = rng.uniform(0, 14)
            health = rng.uniform(.05, 1)
            dash_ready = rng.choice((0, 1))
            stamina = rng.random()
            wall_distance = rng.uniform(.2, ARENA)
            orb_distance = rng.uniform(.5, 30)
            allies = rng.randint(0, 6)
            level = rng.randint(1, 12)

            if stamina < .18 or (
                distance < 4 and health > .55 and dash_ready
            ):
                action = RETREAT
            elif level >= 3 and orb_distance < 4.5 and distance > 7:
                action = GUARD
            elif (
                level >= 2
                and player_speed > 5
                and distance < 17
                and closing_speed < 6
            ):
                action = INTERCEPT
            elif level >= 4 and allies >= 2 and 5 < distance < 19:
                action = FLANK
            elif distance > 22 and player_speed < 3 and wall_distance > 3:
                action = PATROL
            else:
                action = CHASE

            rows.append({
                'distancia': distance,
                'velocidad_cierre': closing_speed,
                'velocidad_jugador': player_speed,
                'vida_jugador': health,
                'dash_disponible': dash_ready,
                'resistencia_dron': stamina,
                'distancia_pared': wall_distance,
                'distancia_energia': orb_distance,
                'aliados_cercanos': allies,
                'nivel': level,
                'accion': action,
            })
        return rows

    def update_ai_decisions(self):
        """Realiza una prediccion por lote en lugar de una por dron y frame."""
        if not self.drones:
            return

        player_pos = self.player.getPos()
        player_speed = self.velocity.length()
        rows = []
        for drone in self.drones:
            node = drone['node']
            position = node.getPos()
            to_player = player_pos - position
            distance = to_player.length()
            direction = normalized(to_player)
            relative_velocity = drone['velocity'] - self.velocity
            closing_speed = relative_velocity.dot(direction)
            wall_distance = ARENA - max(abs(position.x), abs(position.y))
            orb_position = self.nearest_orb_position(position)
            orb_distance = (
                (orb_position - position).length()
                if orb_position is not None
                else 30.0
            )
            allies = sum(
                1
                for other in self.drones
                if other is not drone
                and (other['node'].getPos() - position).lengthSquared() < 36
            )
            rows.append({
                'distancia': distance,
                'velocidad_cierre': closing_speed,
                'velocidad_jugador': player_speed,
                'vida_jugador': self.health / 100,
                'dash_disponible': int(self.dash_cd <= 0),
                'resistencia_dron': drone['stamina'],
                'distancia_pared': max(0, wall_distance),
                'distancia_energia': orb_distance,
                'aliados_cercanos': allies,
                'nivel': self.level,
            })

        predictions = self.ai.predict(pd.DataFrame(rows, columns=AI_FEATURES))
        for drone, prediction in zip(self.drones, predictions):
            mode = int(prediction)
            if mode == INTERCEPT and self.level < 2:
                mode = CHASE
            elif mode == GUARD and self.level < 3:
                mode = PATROL
            elif mode == FLANK and self.level < 4:
                mode = CHASE
            drone['mode'] = mode

    def player_learning_features(self):
        player_pos = self.player.getPos()
        orb_pos = self.nearest_orb_position(player_pos)
        drone_pos = None
        if self.drones:
            drone_pos = min(
                (drone['node'].getPos() for drone in self.drones),
                key=lambda pos: (pos - player_pos).lengthSquared(),
            )
        orb_delta = (orb_pos - player_pos) if orb_pos is not None else Vec3()
        drone_delta = (
            (drone_pos - player_pos) if drone_pos is not None else Vec3()
        )
        return [
            player_pos.x / ARENA,
            player_pos.y / ARENA,
            self.velocity.x / 30,
            self.velocity.y / 30,
            orb_delta.x / (ARENA * 2),
            orb_delta.y / (ARENA * 2),
            drone_delta.x / (ARENA * 2),
            drone_delta.y / (ARENA * 2),
            float(self.dash_cd <= 0),
        ]

    def player_direction_label(self):
        if self.velocity.lengthSquared() < 2.25:
            return 0
        if abs(self.velocity.y) >= abs(self.velocity.x):
            return 1 if self.velocity.y > 0 else 2
        return 3 if self.velocity.x < 0 else 4

    def update_player_model(self, dt):
        self.player_learning_timer -= dt
        if self.player_learning_timer > 0:
            return
        self.player_learning_timer = .25
        current_features = self.player_learning_features()
        if self.previous_player_features is not None:
            label = self.player_direction_label()
            if self.player_samples == 0:
                self.player_model.partial_fit(
                    [self.previous_player_features],
                    [label],
                    classes=[0, 1, 2, 3, 4],
                )
            else:
                self.player_model.partial_fit(
                    [self.previous_player_features],
                    [label],
                )
            self.player_samples += 1
        self.previous_player_features = current_features

    def predicted_player_velocity(self):
        if self.player_samples < 16:
            return Vec3(self.velocity)
        label = int(self.player_model.predict([self.player_learning_features()])[0])
        speed = max(7.0, min(11.0, self.velocity.length()))
        directions = {
            0: Vec3(),
            1: Vec3(0, speed, 0),
            2: Vec3(0, -speed, 0),
            3: Vec3(-speed, 0, 0),
            4: Vec3(speed, 0, 0),
        }
        learned = directions[label]
        return learned * .65 + self.velocity * .35

    def build_world(self):
        self.scene = self.render.attachNewNode('arena')
        self.obstacles = []
        cube(self.scene, (0, 0, -.35), (62, 62, .6), (.018, .027, .075, 1))
        # La imagen del escenario se usa como una gran placa horizontal y la
        # geometría original queda encima para conservar profundidad y colisiones.
        environment_path = Path(__file__).with_name('ecenario_1.jpg')
        if environment_path.exists():
            environment = self.loader.loadTexture(
                Filename.fromOsSpecific(str(environment_path))
            )
            if environment:
                self.add_image_card(
                    self.scene, environment, 50, 50, (0, 0, -.045), horizontal=True,
                    name='environment_art',
                )
        # Pavimento de placas con juntas y circuitos luminosos.
        for position in range(-24, 25, 4):
            cube(
                self.scene,
                (position, 0, -.035),
                (.035, 48, .018),
                (.025, .24, .34, 1),
            )
            cube(
                self.scene,
                (0, position, -.033),
                (48, .035, .018),
                (.025, .24, .34, 1),
            )
        for position in range(-20, 21, 8):
            cube(self.scene, (position, 0, .006), (.075, 48, .025), (.02, .72, .92, 1))
            cube(self.scene, (0, position, .008), (48, .075, .025), (.02, .72, .92, 1))
        for side in (-1, 1):
            # Marcos de borde y paneles segmentados de contencion.
            cube(
                self.scene,
                (side * 25, 0, 1.5),
                (.65, 50, 3),
                (.025, .055, .16, 1),
            )
            cube(
                self.scene,
                (0, side * 25, 1.5),
                (50, .65, 3),
                (.025, .055, .16, 1),
            )
            cube(self.scene, (side * 24.63, 0, 2.65), (.10, 49, .10), (.08, .82, 1, 1))
            cube(self.scene, (0, side * 24.63, 2.65), (49, .10, .10), (.08, .82, 1, 1))
            for p in range(-20, 21, 5):
                cube(self.scene, (side * 24.62, p, 1.5), (.12, .09, 2.45), (.8, .08, .55, 1))
                cube(self.scene, (p, side * 24.62, 1.5), (.09, .12, 2.45), (.8, .08, .55, 1))

        # Paneles de pared y líneas cian repiten las juntas del pavimento.
        wall_panel = (.018, .045, .105, 1)
        wall_glow = (.025, .42, .62, 1)
        wall_accent = (.36, .06, .52, 1)
        for segment in range(-22, 23, 4):
            # Tramos oscuros con bordes metálicos en las cuatro paredes.
            for side in (-1, 1):
                cube(self.scene, (side * 24.57, segment, 1.5),
                     (.12, 3.65, 2.68), wall_panel)
                cube(self.scene, (segment, side * 24.57, 1.5),
                     (3.65, .12, 2.68), wall_panel)
                cube(self.scene, (side * 24.49, segment, 1.5),
                     (.045, .045, 2.52), wall_glow)
                cube(self.scene, (segment, side * 24.49, 1.5),
                     (.045, .045, 2.52), wall_glow)
            # Acentos violetas cada tres paneles para continuar el código neón.
            if segment % 12 == 0:
                for side in (-1, 1):
                    cube(self.scene, (side * 24.48, segment, 1.65),
                         (.05, 1.25, .07), wall_accent)
                    cube(self.scene, (segment, side * 24.48, 1.65),
                         (1.25, .05, .07), wall_accent)
        for side in (-1, 1):
            for z in (.30, 1.48, 2.68):
                cube(self.scene, (side * 24.47, 0, z),
                     (.055, 49, .055), wall_glow)
                cube(self.scene, (0, side * 24.47, z),
                     (49, .055, .055), wall_glow)

        # Gradas neon con el elenco de referencia integrado en la arena.
        self.build_reference_crowd()
        for side in (-1, 1):
            for y, message, color in ((-12, '¡VAMOS EQUIPO!', (.03, .5, .8, 1)), (0, 'NEON ARENA', (.55, .08, .45, 1)), (12, '¡GRAN VUELO!', (.05, .5, .3, 1))):
                cube(self.scene, (side * 25.9, y, 4.2), (.18, 5.2, 1.35), color)
                label = TextNode('arena_banner')
                label.setText(message)
                label.setTextColor(1, 1, .9, 1)
                label.setAlign(TextNode.ACenter)
                sign = self.scene.attachNewNode(label)
                sign.setPos(side * 25.78, y, 4.0)
                sign.setScale(.38)
                sign.setBillboardPointEye()

        world_rng = random.Random(15)
        self.tower_flames = []
        for _ in range(28):
            x = world_rng.uniform(-21, 21)
            y = world_rng.uniform(-21, 21)
            if abs(x) < 7 and abs(y) < 7:
                continue
            height = world_rng.uniform(1.5, 5)
            cube(
                self.scene,
                (x, y, height / 2),
                (1.12, 1.12, height),
                (.025, .075, .20, 1),
            )
            cube(self.scene, (x, y, height * .52), (1.24, 1.24, .12), (.05, .36, .62, 1))
            for offset in (-.43, .43):
                cube(self.scene, (x + offset, y, height / 2), (.055, 1.16, height * .82), (.04, .58, .9, 1))
                cube(self.scene, (x, y + offset, height / 2), (1.16, .055, height * .82), (.65, .06, .48, 1))
            cube(
                self.scene,
                (x, y, height + .08),
                (1.32, 1.32, .2),
                (.05, .9, 1, 1),
            )
            energy_sphere(self.scene, (x, y, height + .32), .22, (.24, 1, .84, 1), 'pillar_core')

            # Llamas geométricas rojo/naranja con núcleo amarillo sobre cada torre.
            flame_root = self.scene.attachNewNode('tower_fire')
            base_z = height + .30
            flame_root.setPos(x, y, base_z)
            flame_vertices = [
                (-.28, -.10, 0), (.28, -.10, 0), (-.17, -.06, .48),
                (.17, -.06, .48), (0, -.025, .98),
                (-.28, .10, 0), (.28, .10, 0), (-.17, .06, .48),
                (.17, .06, .48), (0, .025, .98),
            ]
            flame_faces = [
                (0, 1, 2), (1, 3, 2), (2, 3, 4),
                (5, 7, 6), (6, 7, 8), (7, 9, 8),
                (0, 5, 6), (0, 6, 1), (1, 6, 8), (1, 8, 3),
                (3, 8, 9), (3, 9, 4), (4, 9, 7), (4, 7, 2),
                (2, 7, 5), (2, 5, 0),
            ]
            fire_outer = faceted_mesh(
                flame_root, 'fire_outer', flame_vertices, flame_faces,
                (1, .12, .018, .96),
            )
            fire_inner = faceted_mesh(
                flame_root, 'fire_inner',
                [(-.14, -.12, .03), (.14, -.12, .03), (-.07, -.08, .36),
                 (.07, -.08, .36), (0, -.045, .70),
                 (-.14, .12, .03), (.14, .12, .03), (-.07, .08, .36),
                 (.07, .08, .36), (0, .045, .70)],
                flame_faces, (1, .72, .06, 1),
            )
            fire_outer.setLightOff(1)
            fire_outer.setTransparency(TransparencyAttrib.MAlpha)
            fire_inner.setLightOff(1)
            self.tower_flames.append({
                'node': flame_root,
                'base_z': base_z,
                'phase': world_rng.uniform(0, math.tau),
            })
            self.obstacles.append((x, y, .68, .68))

        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor(Vec4(.22, .28, .48, 1))
        self.render.setLight(self.render.attachNewNode(ambient_light))
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(.45, .68, 1, 1))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(-35, -55, 0)
        self.render.setLight(sun_node)

        self.player = self.scene.attachNewNode('player')
        self.build_drone_model(
            self.player,
            body_color=(.025, .68, .88, 1),
            trim_color=(.2, .96, 1, 1),
            cockpit_color=(.08, .2, .34, 1),
        )

        if self.camera is not None:
            self.camera.setPos(0, -22, 19)
            self.camera.lookAt(0, 0, 0)

    def build_hud(self):
        self.hud = OnscreenText(
            text='',
            pos=(-1.28, .92),
            align=TextNode.ALeft,
            scale=.046,
            fg=(.65, .95, 1, 1),
            mayChange=True,
        )
        self.message = OnscreenText(
            text='',
            pos=(0, .12),
            scale=.095,
            fg=(1, .85, .35, 1),
            mayChange=True,
        )

        # Barra de estado anclada al borde inferior. El fondo se extiende
        # fuera de la pantalla para conservar el ancho completo en 4:3,
        # 16:9 y formatos ultrapanoramicos.
        self.bottom_bar = DirectFrame(
            parent=self.a2dBottomCenter,
            frameColor=(.006, .018, .052, .96),
            frameSize=(-3.0, 3.0, 0.0, .22),
            relief=1,
        )
        DirectFrame(
            parent=self.bottom_bar,
            frameColor=(.06, .8, 1, 1),
            frameSize=(-3.0, 3.0, -.005, .005),
            pos=(0, 0, .215),
        )
        for separator_x in (-.62, .57, 1.40):
            DirectFrame(
                parent=self.bottom_bar,
                frameColor=(.18, .46, .62, .55),
                frameSize=(-.002, .002, -.055, .055),
                pos=(separator_x, 0, .15),
            )

        self.health_label = DirectLabel(
            parent=self.bottom_bar,
            text='VIDA 100%',
            scale=.029,
            pos=(-1.25, 0, .17),
            text_align=TextNode.ALeft,
            text_fg=(.2, 1, .72, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.health_meter = DirectWaitBar(
            parent=self.bottom_bar,
            text='',
            range=100,
            value=100,
            frameSize=(-.28, .28, -.018, .018),
            frameColor=(.018, .075, .12, 1),
            barColor=(.08, .92, .68, 1),
            pos=(-.97, 0, .12),
        )
        self.med_cross_icon = DirectFrame(
            parent=self.bottom_bar,
            frameColor=(0, 0, 0, 0),
            frameSize=(-.09, .09, -.08, .08),
            pos=(-1.65, 0, .145),
        )
        DirectFrame(parent=self.med_cross_icon, frameColor=(1, .95, .95, 1),
                    frameSize=(-.07, .07, -.025, .025))
        DirectFrame(parent=self.med_cross_icon, frameColor=(1, .95, .95, 1),
                    frameSize=(-.025, .025, -.07, .07))
        DirectFrame(parent=self.med_cross_icon, frameColor=(1, .035, .1, 1),
                    frameSize=(-.06, .06, -.016, .016), pos=(0, 0, .012))
        DirectFrame(parent=self.med_cross_icon, frameColor=(1, .035, .1, 1),
                    frameSize=(-.016, .016, -.06, .06), pos=(0, 0, .012))
        self.bottom_cross_count = DirectLabel(
            parent=self.bottom_bar, text='x4', scale=.026,
            pos=(-1.43, 0, .145), text_fg=(1, .43, .48, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.bottom_match_status = DirectLabel(
            parent=self.bottom_bar,
            text='PREPARANDO PARTIDA',
            scale=.026,
            pos=(.04, 0, .145),
            text_fg=(.82, .94, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.bottom_dash_status = DirectLabel(
            parent=self.bottom_bar,
            text='DASH LISTO',
            scale=.026,
            pos=(.66, 0, .145),
            text_fg=(.22, 1, .72, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.bottom_web_status = DirectLabel(
            parent=self.bottom_bar,
            text='TELA 4/6',
            scale=.022,
            pos=(1.12, 0, .145),
            text_fg=(.62, .92, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.spider_icon = DirectFrame(
            parent=self.bottom_bar,
            frameColor=(0, 0, 0, 0),
            frameSize=(-.10, .10, -.07, .07),
            pos=(.93, 0, .15),
        )
        spider_neon = (.18, 1, .76, 1)
        DirectFrame(parent=self.spider_icon, frameColor=spider_neon,
                    frameSize=(-.012, .012, -.017, .017), pos=(0, 0, -.006))
        DirectFrame(parent=self.spider_icon, frameColor=spider_neon,
                    frameSize=(-.019, .019, -.025, .025), pos=(0, 0, -.042))
        DirectFrame(parent=self.spider_icon, frameColor=(.5, 1, .92, 1),
                    frameSize=(-.012, .012, -.012, .012), pos=(0, 0, .03))
        for side in (-1, 1):
            for z_offset, roll in ((.018, 32), (-.012, -32), (-.040, 32), (-.060, -32)):
                leg = DirectFrame(
                    parent=self.spider_icon, frameColor=spider_neon,
                    frameSize=(-.034, .034, -.004, .004),
                    pos=(side * .025, 0, z_offset),
                )
                leg.setR(side * roll)
        self.web_reload_bar = DirectWaitBar(
            parent=self.bottom_bar, text='', range=1, value=1,
            frameSize=(-.11, .11, -.008, .008),
            frameColor=(.025, .09, .12, 1), barColor=spider_neon,
            pos=(.94, 0, .085),
        )
        self.bottom_audio_status = DirectLabel(
            parent=self.bottom_bar,
            text='AUDIO --',
            scale=.026,
            pos=(1.51, 0, .145),
            text_fg=(.62, .72, .82, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.help = DirectLabel(
            parent=self.bottom_bar,
            text=(
                'MANTEN W PARA VOLAR    |    FLECHAS PARA DIRIGIR EN X/Y    |    ESPACIO  DASH    |    F  TELARAÑA    |    '
                'P  PAUSA    |    M  SONIDO    |    R  REINICIAR    |    ESC  SALIR'
            ),
            scale=.024,
            pos=(0, 0, .048),
            text_fg=(.58, .77, .9, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.bottom_bar.hide()

        # Controles visibles durante la partida, en el borde superior derecho.
        self.game_controls = DirectFrame(
            parent=self.aspect2d,
            frameColor=(.006, .018, .052, .88),
            frameSize=(-.60, .60, -.105, .105),
            pos=(.92, 0, .91),
            relief=1,
        )
        self.pause_button = DirectButton(
            parent=self.game_controls,
            text='PAUSA', text_scale=.027,
            text_fg=(.45, .96, 1, 1),
            frameColor=(.025, .18, .28, 1),
            frameSize=(-.17, .17, -.065, .065),
            pos=(-.39, 0, 0), relief=1,
            command=self.toggle_pause,
            rolloverSound=None, clickSound=None,
        )
        DirectButton(
            parent=self.game_controls,
            text='FINALIZAR', text_scale=.023,
            text_fg=(1, .45, .7, 1),
            frameColor=(.24, .035, .14, 1),
            frameSize=(-.20, .20, -.065, .065),
            pos=(0, 0, 0), relief=1,
            command=self.finish_game,
            rolloverSound=None, clickSound=None,
        )
        DirectButton(
            parent=self.game_controls,
            text='REINICIAR', text_scale=.023,
            text_fg=(.68, 1, .78, 1),
            frameColor=(.025, .20, .12, 1),
            frameSize=(-.20, .20, -.065, .065),
            pos=(.39, 0, 0), relief=1,
            command=self.restart_game,
            rolloverSound=None, clickSound=None,
        )
        self.game_controls.hide()

    def toggle_pause(self):
        """Pausa o reanuda la simulacion sin alterar su estado."""
        if not self.game_started or self.finished:
            return
        self.paused = not self.paused
        self.pause_button['text'] = 'SEGUIR' if self.paused else 'PAUSA'
        if self.paused:
            for sound in self.sounds.values():
                sound.stop()
            if self.ambient is not None:
                self.ambient.stop()
            self.message.setText('PAUSADO\nP o el boton SEGUIR para continuar')
            self.message_timer = 0
        else:
            self.message.setText('')
            self.play_ambient()
            self.play_drone_hum()

    def load_best_score(self):
        if not CSV_PATH.exists():
            return 0
        try:
            history = pd.read_csv(CSV_PATH)
            if 'puntuacion' not in history:
                return 0
            scores = pd.to_numeric(history['puntuacion'], errors='coerce').dropna()
            return int(scores.max()) if not scores.empty else 0
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return 0

    def show_message(self, text, duration=2.0):
        self.message.setText(text)
        self.message_timer = duration

    def restart_game(self):
        for shot in getattr(self, 'web_shots', []):
            shot['node'].removeNode()
        self.web_shots = []
        for drone in getattr(self, 'drones', []):
            drone['node'].removeNode()
        for orb in getattr(self, 'orbs', []):
            orb['node'].removeNode()
        if not hasattr(self, 'player'):
            raise RuntimeError('El jugador no fue creado: revisa build_world()')

        for key in self.keys:
            self.keys[key] = False
        for sound in self.sounds.values():
            sound.stop()

        self.player.setPos(0, 0, 0)
        self.flight_height = 0.0
        # El morro del modelo apunta a -Y en local; 180 grados lo orienta a +Y.
        self.player.setH(180)
        self.player.clearColorScale()
        self.velocity = Vec3()
        self.facing_direction = Vec3(0, 1, 0)
        self.health = 100
        self.score = 0
        self.level = 1
        self.elapsed = 0.0
        self.dash_cd = 0.0
        self.web_cooldown = 0.0
        self.web_ammo = 4
        self.dash_timer = 0.0
        self.dash_requested = False
        self.dash_direction = Vec3(0, 1, 0)
        self.hit_cd = 0.0
        self.hit_flash_timer = 0.0
        self.camera_shake = 0.0
        self.finished = False
        self.paused = False
        if hasattr(self, 'pause_button'):
            self.pause_button['text'] = 'PAUSA'
        self.message_timer = 0.0
        self.ai_timer = 0.0
        self.player_learning_timer = 0.0
        self.previous_player_features = None
        self.energy_collected = 0
        self.hits_taken = 0
        self.dash_count = 0
        self.drone_neutralized = 0
        self.max_drones = 0
        self.session_events = []
        self.stats_error = ''
        self.drones = []
        self.orbs = []

        for _ in range(4):
            self.spawn_orb()
        for _ in range(3):
            self.spawn_drone()
        self.max_drones = len(self.drones)

        self.play_ambient(restart=True)
        self.play_drone_hum()
        self.play_sound('start')
        self.show_message('RECOGE ENERGIA Y SOBREVIVE', 3.0)
        self.update_hud()

    def is_blocked(self, x, y, radius=.7):
        if abs(x) > ARENA - 1 or abs(y) > ARENA - 1:
            return True
        return any(
            abs(x - obstacle_x) < half_x + radius
            and abs(y - obstacle_y) < half_y + radius
            for obstacle_x, obstacle_y, half_x, half_y in self.obstacles
        )

    def random_free_position(self, min_player_distance=4.0, clearance=.8):
        for _ in range(80):
            x = self.rng.uniform(-19, 19)
            y = self.rng.uniform(-19, 19)
            if self.is_blocked(x, y, clearance):
                continue
            candidate = Vec3(x, y, 0)
            if (
                hasattr(self, 'player')
                and (candidate - self.player.getPos()).length() < min_player_distance
            ):
                continue
            occupied = [orb['node'].getPos() for orb in getattr(self, 'orbs', [])]
            occupied.extend(
                drone['node'].getPos() for drone in getattr(self, 'drones', [])
            )
            if any((candidate - position).length() < 2.2 for position in occupied):
                continue
            return x, y

        for x, y in ((-18, -18), (18, 18), (-18, 18), (18, -18), (0, 15)):
            if not self.is_blocked(x, y, clearance):
                return x, y
        return 0, 10

    def spawn_orb(self):
        x, y = self.random_free_position(min_player_distance=3.5, clearance=.8)
        node = self.scene.attachNewNode('red_cross_pickup')
        node.setPos(x, y, .8)
        # Cruz de primeros auxilios 3D, con ribete blanco y núcleo rojo visible.
        white = (.9, .96, 1, 1)
        red = (1, .035, .10, 1)
        cube(node, (0, 0, 0), (1.12, .40, .18), white)
        cube(node, (0, 0, 0), (.40, 1.12, .18), white)
        cube(node, (0, -.015, .105), (.94, .23, .16), red)
        cube(node, (0, -.015, .106), (.23, .94, .16), red)
        self.orbs.append({
            'node': node,
            'phase': self.rng.uniform(0, math.tau),
        })

    def spawn_drone(self, elite=False):
        x, y = self.random_free_position(min_player_distance=13, clearance=1.2)
        node = self.scene.attachNewNode('elite_drone' if elite else 'enemy_drone')
        node.setPos(x, y, 0)
        size = 1.25 if elite else 1.0
        variant = 'purple' if elite else self.rng.choice(('green', 'red'))
        palette = {
            'purple': ((.24, .12, .34, 1), (1, .18, .78, 1), (.16, .22, .38, 1)),
            'green': ((.08, .30, .20, 1), (.12, 1, .52, 1), (.10, .28, .22, 1)),
            'red': ((.36, .10, .13, 1), (1, .16, .30, 1), (.34, .12, .17, 1)),
        }
        body_color, trim_color, cockpit_color = palette[variant]
        self.build_drone_model(
            node,
            body_color=body_color,
            trim_color=trim_color,
            cockpit_color=cockpit_color,
            size=size,
            variant=variant,
        )
        self.drones.append({
            'node': node,
            'base_speed': self.rng.uniform(3.25, 4.25) + (.75 if elite else 0),
            'velocity': Vec3(),
            'stamina': self.rng.uniform(.65, 1),
            'mode': CHASE,
            'flank_sign': self.rng.choice((-1, 1)),
            'elite': elite,
            'webbed_until': 0.0,
            'web_node': None,
        })

    @staticmethod
    def add_image_card(parent, texture, width, height, position, horizontal=False,
                       name='image_card'):
        """Crea una tarjeta texturizada para arte 2D dentro de la escena 3D."""
        maker = CardMaker(name)
        maker.setFrame(-width / 2, width / 2, 0, height)
        card = parent.attachNewNode(maker.generate())
        card.setPos(*position)
        if horizontal:
            card.setP(-90)
        card.setTexture(texture, 1)
        card.setTwoSided(True)
        card.setLightOff(1)
        return card

    def build_reference_crowd(self):
        """Coloca los cuatro personajes de referencia en gradas visibles."""
        self.reference_crowd = []
        asset_dir = Path(__file__).parent / 'output' / 'imagegen'
        asset_names = (
            'npc_publico_arabe_halcon.png',
            'npc_publico_magnate_dinero.png',
            'npc_publico_magnate_maletin.png',
            'npc_publico_mujer_glamour.png',
        )
        textures = []
        for asset_name in asset_names:
            asset_path = asset_dir / asset_name
            if not asset_path.exists():
                continue
            texture = self.loader.loadTexture(
                Filename.fromOsSpecific(str(asset_path))
            )
            if texture:
                textures.append(texture)

        if not textures:
            return

        # Extend the stands outside the combat floor; the wall stays clear.
        stand_rows = ((26.5, 3.08, 2.75, 2.84), (29.15, 4.10, 2.25, 2.28))
        positions = (-18, -12, -6, 0, 6, 12, 18)
        for axis, side in (('x', -1), ('x', 1), ('y', -1), ('y', 1)):
            for row_index, (distance, base_z, height, tier_width) in enumerate(stand_rows):
                if axis == 'x':
                    deck_pos = (side * distance, 0, base_z - .39)
                    deck_scale = (tier_width, 49, .42)
                    rail_pos = (side * distance, 0, base_z - .12)
                    rail_scale = (.075, 49, .07)
                else:
                    deck_pos = (0, side * distance, base_z - .39)
                    deck_scale = (49, tier_width, .42)
                    rail_pos = (0, side * distance, base_z - .12)
                    rail_scale = (49, .075, .07)
                cube(self.scene, deck_pos, deck_scale, (.025, .045, .12, 1))
                cube(self.scene, rail_pos, rail_scale, (.035, .55, .78, 1))

                card_width = height * 2 / 3
                for slot, along in enumerate(positions):
                    if axis == 'x':
                        position = (side * distance, along, base_z)
                    else:
                        position = (along, side * distance, base_z)
                    texture_index = (slot + row_index + (side > 0)) % len(textures)
                    actor = self.scene.attachNewNode('reference_crowd_npc')
                    actor.setPos(*position)
                    card = self.add_image_card(
                        actor, textures[texture_index], card_width, height,
                        (0, 0, 0), name='reference_character_card',
                    )
                    card.setTransparency(TransparencyAttrib.MAlpha)
                    card.setDepthWrite(False)
                    card.setBin('transparent', 20)
                    card.setBillboardPointEye()
                    self.reference_crowd.append({
                        'node': actor,
                        'base_z': base_z,
                        'phase': (slot * 1.7 + row_index + side) * .7,
                    })

    @staticmethod
    def build_drone_model(parent, body_color, trim_color, cockpit_color, size=1.0,
                          variant='player'):
        """Construye un caza dron low-poly 3D inspirado en las imágenes adjuntas."""
        def point(x, y, z):
            return (x * size, y * size, z * size)

        # Casco central afilado con morro frontal, lomo elevado y volumen real.
        bottom = [
            point(0, -1.42, 1.10), point(-.34, -.72, 1.03),
            point(-.40, .48, 1.04), point(0, 1.03, 1.08),
            point(.40, .48, 1.04), point(.34, -.72, 1.03),
        ]
        rim = [
            point(0, -1.42, 1.25), point(-.34, -.72, 1.48),
            point(-.32, .46, 1.40), point(0, .94, 1.24),
            point(.32, .46, 1.40), point(.34, -.72, 1.48),
        ]
        ridge = point(0, -.20, 1.78)
        vertices = bottom + rim + [ridge]
        triangles = []
        for index in range(6):
            nxt = (index + 1) % 6
            triangles.extend(((index, nxt, 6 + nxt), (index, 6 + nxt, 6 + index)))
            triangles.append((6 + index, 6 + nxt, 12))
        triangles.extend(((0, 5, 4), (0, 4, 3), (0, 3, 2), (0, 2, 1)))
        faceted_mesh(parent, 'drone_armored_fuselage', vertices, triangles, body_color)

        # Alas delta barridas, construidas como prismas facetados a ambos lados.
        for side in (-1, 1):
            outline = [
                point(side * .27, -.68, 1.30),
                point(side * 1.98, -.02, 1.18),
                point(side * 1.43, .86, 1.13),
                point(side * .34, .60, 1.31),
            ]
            lower = [(x, y, z - .16 * size) for x, y, z in outline]
            wing_vertices = outline + lower
            wing_triangles = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
            for index in range(4):
                nxt = (index + 1) % 4
                wing_triangles.extend(((index, nxt, 4 + nxt), (index, 4 + nxt, 4 + index)))
            faceted_mesh(parent, f'delta_wing_{side}', wing_vertices, wing_triangles, body_color)

            # Nervaduras luminosas sobre las alas y pods laterales blindados.
            stripe = cube(
                parent, point(side * 1.08, .02, 1.30),
                (.075 * size, 1.10 * size, .055 * size), trim_color,
            )
            stripe.setH(side * 53)
            pod = energy_sphere(
                parent, point(side * 1.52, .48, 1.09), .28 * size,
                cockpit_color, 'wing_engine_pod',
            )
            apply_metal_finish(pod, cockpit_color, 58)
            pod.setScale(.75 * size, 1.25 * size, .72 * size)
            energy_sphere(
                parent, point(side * 1.52, .70, 1.10), .12 * size,
                trim_color, 'wing_engine_glow',
            ).setScale(.8 * size, 1.5 * size, .8 * size)

        # Cabina facetada azul oscura, con visor y luces de navegación.
        canopy = energy_sphere(
            parent, point(0, -.23, 1.57), .34 * size,
            cockpit_color, 'armored_cockpit',
        )
        apply_metal_finish(canopy, cockpit_color, 88)
        canopy.setScale(.78 * size, 1.75 * size, .60 * size)
        canopy.setH(0)
        energy_sphere(
            parent, point(0, -.90, 1.42), .095 * size,
            trim_color, 'nose_beacon',
        ).setScale(.72 * size, .65 * size, .72 * size)
        for side in (-1, 1):
            energy_sphere(
                parent, point(side * .25, -.55, 1.48), .07 * size,
                trim_color, 'navigation_light',
            )

        # Tres motores traseros con núcleo brillante y estela de escape.
        for x in (-.48, 0, .48):
            nozzle = energy_sphere(
                parent, point(x, .73, 1.05), .20 * size,
                cockpit_color, 'rear_thruster',
            )
            apply_metal_finish(nozzle, cockpit_color, 76)
            nozzle.setScale(.88 * size, 1.20 * size, .88 * size)
            exhaust = energy_sphere(
                parent, point(x, 1.05, .96), .115 * size,
                trim_color, 'thruster_glow',
            )
            exhaust.setScale(.8 * size, 1.8 * size, .8 * size)

        # Aletas de cola y marcas de blindaje para reforzar la silueta 3D.
        for side in (-1, 1):
            fin = cube(
                parent, point(side * .28, .65, 1.42),
                (.12 * size, .62 * size, .42 * size), trim_color,
            )
            apply_metal_finish(fin, trim_color, 54)
            fin.setH(side * 20)

        # La referencia muestra un rostro frontal vertical, no solo un ala.
        # Añadimos una coraza alta facetada sobre el fuselaje para formar ese
        # volumen y hacer visible el sensor central desde la cámara del juego.
        chest = cube(parent, point(0, -.60, 1.28),
                     (.86 * size, .26 * size, 1.35 * size), body_color)
        apply_metal_finish(chest, body_color, 72)
        brow = cube(parent, point(0, -.77, 1.90),
                    (.72 * size, .16 * size, .22 * size), cockpit_color)
        apply_metal_finish(brow, cockpit_color, 86)

        # Ojo/energía circular con anillos tridimensionales concéntricos.
        socket = energy_sphere(parent, point(0, -.83, 1.42), .48 * size,
                               cockpit_color, 'drone_eye_socket')
        socket.setScale(size, .26 * size, size)
        apply_metal_finish(socket, cockpit_color, 96)
        for radius, count, bead_size in ((.36, 16, .045), (.27, 12, .032)):
            for index in range(count):
                angle = math.tau * index / count
                bead = energy_sphere(
                    parent,
                    point(math.cos(angle) * radius, -.99,
                          1.42 + math.sin(angle) * radius),
                    bead_size * size, trim_color, 'eye_ring_segment',
                )
                bead.setScale(size, .56 * size, size)
        eye = energy_sphere(parent, point(0, -1.01, 1.42), .18 * size,
                            trim_color, 'eye_luminous_core')
        eye.setScale(size, .52 * size, size)
        eye_core = energy_sphere(parent, point(0, -1.04, 1.42), .085 * size,
                                 (1, .96, .82, 1), 'eye_hotspot')
        eye_core.setScale(size, .45 * size, size)

        # Aletas verticales angulares a los lados; el rojo lleva puntas extra.
        for side in (-1, 1):
            for index, (anchor_x, anchor_z, tip_x, tip_z, width) in enumerate((
                (.48, 1.38, 1.55, 2.20 if variant != 'green' else 2.05, .60),
                (.52, .96, 1.46, .36, .46),
            )):
                anchor_x *= side
                tip_x *= side
                dx, dz = tip_x - anchor_x, tip_z - anchor_z
                length = max(math.hypot(dx, dz), .001)
                nx, nz = -dz / length, dx / length
                vertices = [
                    point(anchor_x + nx * width / 2, -.36, anchor_z + nz * width / 2),
                    point(anchor_x - nx * width / 2, -.36, anchor_z - nz * width / 2),
                    point(tip_x, -.23, tip_z),
                    point(anchor_x + nx * width / 2, .18, anchor_z + nz * width / 2),
                    point(anchor_x - nx * width / 2, .18, anchor_z - nz * width / 2),
                    point(tip_x, .12, tip_z),
                ]
                faceted_mesh(
                    parent, f'faceted_side_blade_{side}_{index}', vertices,
                    [(0, 1, 2), (4, 3, 5), (0, 2, 5), (0, 5, 3),
                     (1, 4, 5), (1, 5, 2), (0, 3, 4), (0, 4, 1)],
                    body_color,
                )
            if variant == 'red':
                spur = cube(parent, point(side * 1.0, -.29, 1.63),
                            (.15 * size, .18 * size, .95 * size), trim_color)
                spur.setR(side * 36)
                apply_metal_finish(spur, trim_color, 68)

        # Propulsores gemelos inferiores, con tobera metálica y llama de neón.
        for x in (-.53, .53):
            engine = energy_sphere(parent, point(x, .02, .28), .25 * size,
                                   cockpit_color, 'lower_engine_housing')
            engine.setScale(.92 * size, .90 * size, 1.35 * size)
            apply_metal_finish(engine, cockpit_color, 78)
            exhaust = energy_sphere(parent, point(x, .02, -.01), .12 * size,
                                    trim_color, 'lower_engine_glow')
            exhaust.setScale(.78 * size, .78 * size, 1.85 * size)

    def nearest_orb_position(self, origin):
        if not self.orbs:
            return None
        return min(
            (orb['node'].getPos() for orb in self.orbs),
            key=lambda position: (position - origin).lengthSquared(),
        )

    @staticmethod
    def target_drone_count(level):
        return min(3 + level - 1, 8)

    @staticmethod
    def target_orb_count(level):
        return min(6, 4 + (level - 1) // 3)

    def apply_level(self, new_level):
        self.level = new_level
        self.health = min(100, self.health + 10)
        self.web_ammo = min(self.web_max_ammo, self.web_ammo + 1)
        target_count = self.target_drone_count(self.level)
        while len(self.drones) < target_count:
            elite = self.level % 5 == 0 and len(self.drones) == target_count - 1
            self.spawn_drone(elite=elite)
        while len(self.orbs) < self.target_orb_count(self.level):
            self.spawn_orb()
        self.max_drones = max(self.max_drones, len(self.drones))
        self.ai_timer = 0

        unlocks = {
            2: 'DRONES DESBLOQUEAN INTERCEPCION',
            3: 'DRONES PROTEGEN LA ENERGIA',
            4: 'DRONES DESBLOQUEAN FLANQUEO',
            5: 'APARECE UN DRON ELITE',
        }
        detail = unlocks.get(self.level, 'VELOCIDAD Y REACCION AUMENTAN')
        self.show_message(f'NIVEL {self.level}\n{detail}\n+10 VIDA', 2.8)
        self.play_sound('level')
        self.session_events.append({
            'segundo': round(self.elapsed, 2),
            'evento': 'nivel',
            'nivel': self.level,
        })

    def move_player(self, displacement):
        position = self.player.getPos()
        target_x = max(-ARENA + 1, min(ARENA - 1, position.x + displacement.x))
        if not self.is_blocked(target_x, position.y):
            self.player.setX(target_x)
        else:
            self.velocity.x = 0

        position = self.player.getPos()
        target_y = max(-ARENA + 1, min(ARENA - 1, position.y + displacement.y))
        if not self.is_blocked(position.x, target_y):
            self.player.setY(target_y)
        else:
            self.velocity.y = 0

    def update_player(self, dt):
        horizontal = float(self.keys['d'] or self.keys['arrow_right']) - float(
            self.keys['a'] or self.keys['arrow_left']
        )
        vertical = float(self.keys['arrow_up']) - float(
            self.keys['arrow_down']
        )
        w_down = self.keys['w']
        flight_target = 3.4 if w_down else 0.0
        self.flight_height += (flight_target - self.flight_height) * min(1.0, dt * 5.0)
        self.player.setZ(self.flight_height)
        direction = Vec3(horizontal, vertical, 0)
        if direction.lengthSquared() > 0:
            direction.normalize()
            self.facing_direction = Vec3(direction)

        self.dash_cd = max(0.0, self.dash_cd - dt)
        self.dash_timer = max(0.0, self.dash_timer - dt)
        if self.dash_requested:
            if self.dash_cd <= 0:
                self.dash_direction = (
                    Vec3(direction)
                    if direction.lengthSquared() > 0
                    else Vec3(self.facing_direction)
                )
                self.dash_timer = .18
                self.dash_cd = 1.9
                self.dash_count += 1
                self.play_sound('dash')
                self.session_events.append({
                    'segundo': round(self.elapsed, 2),
                    'evento': 'dash',
                    'nivel': self.level,
                })
            self.dash_requested = False

        if self.dash_timer > 0:
            target_velocity = self.dash_direction * 27
            response = 28
        else:
            target_velocity = direction * 9.5
            response = 14 if direction.lengthSquared() > 0 else 8
        self.velocity += (target_velocity - self.velocity) * min(1.0, response * dt)
        self.move_player(self.velocity * dt)

        if self.velocity.lengthSquared() > .3:
            self.player.setH(
                math.degrees(math.atan2(self.velocity.x, -self.velocity.y))
            )

    def update_orbs(self, dt):
        for orb in self.orbs[:]:
            node = orb['node']
            orb['phase'] += dt * 2.8
            pulse = math.sin(orb['phase'])
            node.setZ(.8 + pulse * .18)
            node.setH(node.getH() + 95 * dt)
            size = .8 * (1 + pulse * .07)
            node.setScale(size)

            if (node.getPos() - self.player.getPos()).length() >= 1.75:
                continue

            self.score += 10
            self.energy_collected += 1
            self.health = min(100, self.health + 2)
            self.web_ammo = min(self.web_max_ammo, self.web_ammo + 1)
            node.removeNode()
            self.orbs.remove(orb)
            self.spawn_orb()
            self.play_sound('collect')
            self.show_message('CRUZ ROJA: VIDA +2  |  TELARAÑA +1', 1.1)
            self.session_events.append({
                'segundo': round(self.elapsed, 2),
                'evento': 'energia',
                'nivel': self.level,
            })

            new_level = 1 + self.score // POINTS_PER_LEVEL
            if new_level > self.level:
                self.apply_level(new_level)
            else:
                self.show_message('CRUZ ROJA: VIDA +2  |  TELARAÑA +1', 1.1)

    def obstacle_avoidance(self, position):
        force = Vec3()
        for x, y, _, _ in self.obstacles:
            away = position - Vec3(x, y, 0)
            distance = away.length()
            if .01 < distance < 3.2:
                force += normalized(away) * ((3.2 - distance) / 3.2)
        return force

    def desired_drone_direction(self, drone):
        node = drone['node']
        position = node.getPos()
        player_pos = self.player.getPos()
        to_player = player_pos - position
        direct = normalized(to_player, (0, 1, 0))
        perpendicular = Vec3(-direct.y, direct.x, 0) * drone['flank_sign']
        predicted_velocity = self.predicted_player_velocity()
        mode = drone['mode']

        if mode == PATROL:
            radial = normalized(-position, (0, 1, 0))
            desired = perpendicular * .75 + radial * .35
        elif mode == INTERCEPT:
            lead_time = max(.3, min(1.4, to_player.length() / 12))
            target = player_pos + predicted_velocity * lead_time
            desired = normalized(target - position, direct)
        elif mode == FLANK:
            target = player_pos + predicted_velocity * .35 + perpendicular * 5
            desired = normalized(target - position, direct)
        elif mode == GUARD:
            orb_pos = self.nearest_orb_position(position)
            if orb_pos is None:
                desired = direct
            else:
                guard_point = orb_pos + normalized(player_pos - orb_pos) * 2.2
                offset = guard_point - position
                desired = (
                    normalized(offset, direct)
                    if offset.length() > 2
                    else perpendicular
                )
        elif mode == RETREAT:
            desired = -direct + perpendicular * .35
        else:
            desired = direct

        separation = Vec3()
        for other in self.drones:
            if other is drone:
                continue
            away = position - other['node'].getPos()
            distance = away.length()
            if .01 < distance < 3.3:
                separation += normalized(away) * ((3.3 - distance) / 3.3)

        wall_force = Vec3()
        margin = 4.0
        if position.x > ARENA - margin:
            wall_force.x -= (position.x - (ARENA - margin)) / margin
        elif position.x < -ARENA + margin:
            wall_force.x += (-ARENA + margin - position.x) / margin
        if position.y > ARENA - margin:
            wall_force.y -= (position.y - (ARENA - margin)) / margin
        elif position.y < -ARENA + margin:
            wall_force.y += (-ARENA + margin - position.y) / margin

        desired += separation * 1.25
        desired += wall_force * 1.6
        desired += self.obstacle_avoidance(position) * .85
        return normalized(desired, direct)

    def damage_player(self, drone):
        base_damage = min(16 + 2 * (self.level - 1), 28)
        damage = base_damage + (6 if drone['elite'] else 0)
        self.health = max(0, self.health - damage)
        self.hits_taken += 1
        self.hit_cd = 1.1
        self.hit_flash_timer = .24
        self.camera_shake = .28
        self.player.setColorScale(1, .22, .22, 1)
        self.play_sound('crowd_boo')
        away = normalized(
            self.player.getPos() - drone['node'].getPos(),
            (0, -1, 0),
        )
        self.move_player(away * 1.4)
        self.velocity = away * 8
        self.play_sound('hit')
        self.show_message(f'IMPACTO  -{damage} VIDA', 1.1)
        self.session_events.append({
            'segundo': round(self.elapsed, 2),
            'evento': 'impacto',
            'nivel': self.level,
        })
        if self.health <= 0:
            self.finish_game()

    def update_drones(self, dt):
        speed_multiplier = min(1 + .055 * (self.level - 1), 1.45)
        mode_speed = {
            PATROL: .68,
            CHASE: 1.0,
            INTERCEPT: 1.10,
            FLANK: .96,
            GUARD: .78,
            RETREAT: .84,
        }

        for drone in self.drones[:]:
            if self.finished:
                break
            neutralize_at = drone.get('neutralize_at', 0.0)
            if neutralize_at and self.elapsed >= neutralize_at:
                if drone.get('web_node') is not None:
                    drone['web_node'].removeNode()
                drone['node'].removeNode()
                self.drones.remove(drone)
                self.drone_neutralized += 1
                self.score += 25
                self.web_ammo = min(self.web_max_ammo, self.web_ammo + 1)
                self.play_sound('web_kill')
                self.play_sound('crowd_cheer')
                self.play_sound('crowd_applause')
                self.session_events.append({
                    'segundo': round(self.elapsed, 2),
                    'evento': 'dron_neutralizado',
                    'nivel': self.level,
                    'puntos': 25,
                })
                self.show_message('+25 PUNTOS  |  DRON NEUTRALIZADO', 1.1)
                new_level = 1 + self.score // POINTS_PER_LEVEL
                if new_level > self.level:
                    self.apply_level(new_level)
                else:
                    while len(self.drones) < self.target_drone_count(self.level):
                        self.spawn_drone()
                continue
            if drone.get('webbed_until', 0) > self.elapsed:
                drone['velocity'] = Vec3()
                continue
            if drone.get('web_node') is not None:
                drone['web_node'].removeNode()
                drone['web_node'] = None
                drone['webbed_until'] = 0.0
            desired = self.desired_drone_direction(drone)
            speed = (
                drone['base_speed']
                * speed_multiplier
                * mode_speed[drone['mode']]
                * (.76 + .24 * drone['stamina'])
            )
            target_velocity = desired * speed
            response = 3.5 + min(self.level, 10) * .08
            drone['velocity'] += (
                target_velocity - drone['velocity']
            ) * min(1.0, response * dt)

            if drone['mode'] in (CHASE, INTERCEPT, FLANK):
                drone['stamina'] = max(0, drone['stamina'] - .075 * dt)
            else:
                drone['stamina'] = min(1, drone['stamina'] + .12 * dt)

            node = drone['node']
            next_position = node.getPos() + drone['velocity'] * dt
            clamped_x = max(-ARENA + 1, min(ARENA - 1, next_position.x))
            clamped_y = max(-ARENA + 1, min(ARENA - 1, next_position.y))
            if clamped_x != next_position.x:
                drone['velocity'].x *= -.25
            if clamped_y != next_position.y:
                drone['velocity'].y *= -.25
            target_z = 1.2 + .65 * math.sin(self.elapsed * 2.2 + drone['flank_sign'])
            for other in self.drones:
                if other is drone:
                    continue
                other_pos = other['node'].getPos()
                if math.hypot(clamped_x - other_pos.x, clamped_y - other_pos.y) < 2.8:
                    target_z = max(target_z, other_pos.z + 1.7)
            target_z = min(7.0, target_z)
            next_z = node.getZ() + (target_z - node.getZ()) * min(1.0, dt * 2.5)
            node.setPos(clamped_x, clamped_y, next_z)
            if drone['velocity'].lengthSquared() > .02:
                node.setH(
                    math.degrees(
                        math.atan2(drone['velocity'].x, -drone['velocity'].y)
                    )
                )

            collision_distance = (
                self.player.getPos() - node.getPos()
            ).length()
            collision_radius = 2.1 if drone['elite'] else 1.75
            if collision_distance < collision_radius and abs(self.player.getZ() - node.getZ()) < 1.5 and self.hit_cd <= 0:
                self.damage_player(drone)

    def save_statistics(self):
        row = pd.DataFrame([{
            'fecha': datetime.now().isoformat(timespec='seconds'),
            'puntuacion': self.score,
            'nivel': self.level,
            'segundos': round(self.elapsed, 1),
            'energia_recogida': self.energy_collected,
            'impactos': self.hits_taken,
            'dashes': self.dash_count,
            'drones_neutralizados': self.drone_neutralized,
            'maximo_drones': self.max_drones,
            'precision_ia': round(self.ai_accuracy, 4),
            'muestras_aprendizaje': self.player_samples,
        }])
        try:
            if CSV_PATH.exists() and CSV_PATH.stat().st_size:
                history = pd.read_csv(CSV_PATH)
                history = pd.concat([history, row], ignore_index=True, sort=False)
            else:
                history = row
            history.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
            return True
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            self.stats_error = str(exc)
            return False

    def finish_game(self):
        if self.finished:
            return
        self.finished = True
        self.paused = False
        if hasattr(self, 'pause_button'):
            self.pause_button['text'] = 'PAUSA'
        new_record = self.score > self.best_score
        saved = self.save_statistics()
        self.best_score = max(self.best_score, self.score)
        if self.ambient is not None:
            self.ambient.stop()
        drone_hum = self.sounds.get('drone_hum')
        if drone_hum is not None:
            drone_hum.stop()
        self.play_sound('game_over')
        record_text = ' | NUEVO RECORD' if new_record else ''
        save_text = '' if saved else '\nNO SE PUDO GUARDAR EL CSV'
        self.message.setText(
            f'FIN DEL JUEGO | {self.score} PUNTOS{record_text}'
            f'\nR PARA REINICIAR{save_text}'
        )
        self.message_timer = 0

    def update_hud(self):
        next_level = POINTS_PER_LEVEL - self.score % POINTS_PER_LEVEL
        dash_text = 'LISTO' if self.dash_cd <= 0 else f'{self.dash_cd:.1f}s'
        audio_text = 'ON' if self.audio_enabled and self.audio_available else 'OFF'
        mode_counts = {}
        for drone in self.drones:
            name = BEHAVIOR_NAMES[drone['mode']]
            mode_counts[name] = mode_counts.get(name, 0) + 1
        dominant_mode = (
            max(mode_counts, key=mode_counts.get) if mode_counts else 'SIN DRONES'
        )
        self.hud.setText(
            f'NEON ARENA 3D | {self.game_mode.upper()}\n'
            f'IA TACTICA: {dominant_mode}\n'
            f'VALIDACION RF: {self.ai_accuracy:.1%}\n'
            f'IA JUGADOR: {self.player_samples} muestras\n'
            f'DRONES NEUTRALIZADOS: {self.drone_neutralized}'
        )

        health_value = max(0, min(100, self.health))
        if health_value > 55:
            health_color = (.08, .92, .68, 1)
        elif health_value > 25:
            health_color = (1, .68, .12, 1)
        else:
            health_color = (1, .18, .28, 1)
        self.health_meter['value'] = health_value
        self.health_meter['barColor'] = health_color
        self.health_label['text'] = f'VIDA {health_value:.0f}%'
        self.health_label['text_fg'] = health_color
        self.bottom_cross_count['text'] = f'x{len(self.orbs)}'

        minutes, seconds = divmod(int(self.elapsed), 60)
        self.bottom_match_status['text'] = (
            f'PUNTOS {self.score:04d}  |  RECORD {self.best_score:04d}  |  '
            f'NIVEL {self.level}  |  SIG. {next_level}  |  '
            f'{minutes:02d}:{seconds:02d}  |  DRONES {len(self.drones)}/8'
        )
        self.bottom_dash_status['text'] = f'DASH {dash_text}'
        self.bottom_dash_status['text_fg'] = (
            (.22, 1, .72, 1) if self.dash_cd <= 0 else (1, .7, .2, 1)
        )
        web_ready = self.web_cooldown <= 0 and self.web_ammo > 0
        cooldown_text = 'LISTA' if self.web_cooldown <= 0 else f'{self.web_cooldown:.1f}s'
        self.bottom_web_status['text'] = (
            f'TELA {self.web_ammo}/{self.web_max_ammo} {cooldown_text}'
        )
        self.bottom_web_status['text_fg'] = (
            (.45, 1, .78, 1) if web_ready else (1, .62, .25, 1)
        )
        self.web_reload_bar['value'] = (
            1.0 if self.web_cooldown <= 0
            else 1.0 - self.web_cooldown / 1.65
        )
        self.web_reload_bar['barColor'] = (
            (.18, 1, .72, 1) if web_ready else (1, .48, .18, 1)
        )
        pulse = .72 + .28 * (math.sin(self.elapsed * 7) + 1) / 2
        self.spider_icon.setColorScale(
            pulse, pulse if web_ready else .55, 1 if web_ready else .3, 1
        )
        self.bottom_audio_status['text'] = f'AUDIO {audio_text}'
        self.bottom_audio_status['text_fg'] = (
            (.25, .9, 1, 1)
            if self.audio_enabled and self.audio_available
            else (1, .42, .42, 1)
        )

    def update_camera(self, dt):
        if self.camera is None:
            return
        player_pos = self.player.getPos()
        target = player_pos + Vec3(0, -19, 17)
        self.camera_shake = max(0.0, self.camera_shake - dt)
        if self.camera_shake > 0:
            strength = self.camera_shake * 1.2
            target += Vec3(
                self.rng.uniform(-strength, strength),
                self.rng.uniform(-strength, strength),
                self.rng.uniform(-strength, strength),
            )
        self.camera.setPos(
            self.camera.getPos()
            + (target - self.camera.getPos()) * min(1.0, dt * 4)
        )
        self.camera.lookAt(player_pos + Vec3(0, 0, 1))

    def update_game(self, task):
        dt = min(globalClock.getDt(), .06)
        self.update_tower_flames(dt)
        self.update_crowd(dt)
        if not self.game_started:
            return Task.cont
        if self.paused:
            return Task.cont
        if self.finished:
            return Task.cont

        self.elapsed += dt
        self.hit_cd = max(0.0, self.hit_cd - dt)
        self.hit_flash_timer = max(0.0, self.hit_flash_timer - dt)
        if self.hit_flash_timer <= 0 and self.player.hasColorScale():
            self.player.clearColorScale()

        self.update_player(dt)
        self.update_player_model(dt)

        self.ai_timer -= dt
        if self.ai_timer <= 0:
            self.update_ai_decisions()
            self.ai_timer = max(.10, .30 - .018 * (self.level - 1))

        self.update_orbs(dt)
        self.update_web_shots(dt)
        self.update_drones(dt)
        self.update_camera(dt)

        if not self.finished and self.message_timer > 0:
            self.message_timer = max(0.0, self.message_timer - dt)
            if self.message_timer <= 0:
                self.message.setText('')

        self.update_hud()
        return Task.cont


if __name__ == '__main__':
    NeonArena().run()
