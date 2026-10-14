"""NEON ARENA 3D - Panda3D + pandas + scikit-learn.
Controls: WASD move, SPACE dash, R restart, ESC quit.
Install: python -m pip install panda3d pandas scikit-learn
Run: python neon_arena_3d.py
"""
import math
import random
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    raise SystemExit('Falta una dependencia. Ejecuta: python -m pip install panda3d pandas scikit-learn') from exc

try:
    from direct.showbase.ShowBase import ShowBase
    from direct.showbase.ShowBaseGlobal import globalClock
    from direct.task import Task
    from direct.gui.OnscreenText import OnscreenText
    from panda3d.core import (AmbientLight, DirectionalLight, TextNode, Vec3, Vec4,
                              Geom, GeomNode, GeomTriangles, GeomVertexData,
                              GeomVertexFormat, GeomVertexWriter, WindowProperties)
except ImportError as exc:
    raise SystemExit('Falta Panda3D. Ejecuta: python -m pip install panda3d') from exc

ARENA = 23
CSV_PATH = Path(__file__).with_name('estadisticas_neon_arena.csv')


def cube(parent, pos, scale, rgba):
    """Creates a cube procedurally: no external assets required."""
    vertices = [(-.5,-.5,-.5),(.5,-.5,-.5),(.5,.5,-.5),(-.5,.5,-.5),
                (-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)]
    faces = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
    fmt = GeomVertexFormat.getV3()
    data = GeomVertexData('cube', fmt, Geom.UHStatic)
    writer = GeomVertexWriter(data, 'vertex')
    for x,y,z in vertices:
        writer.addData3(x,y,z)
    tris = GeomTriangles(Geom.UHStatic)
    for a,b,c,d in faces:
        tris.addVertices(a,b,c)
        tris.addVertices(a,c,d)
    geom = Geom(data)
    geom.addPrimitive(tris)
    node = GeomNode('cube')
    node.addGeom(geom)
    item = parent.attachNewNode(node)
    item.setPos(*pos)
    item.setScale(*scale)
    item.setColor(*rgba)
    item.setTwoSided(True)
    return item


class NeonArena(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(.015, .025, .065, 1)
        props = WindowProperties()
        props.setTitle('NEON ARENA 3D | AI BATTLE')
        props.setSize(1280, 720)
        if self.win is not None and hasattr(self.win, 'requestProperties'):
            self.win.requestProperties(props)
        self.keys = {k: False for k in ('w','a','s','d','space')}
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(key+'-up', self.set_key, [key, False])
        self.accept('r', self.restart_game)
        self.accept('escape', self.userExit)
        self.last_time = 0
        self.session_events = []
        self.build_ai()
        self.build_world()
        self.build_hud()
        self.restart_game()
        self.taskMgr.add(self.update_game, 'update_game')

    def set_key(self, key, value):
        self.keys[key] = value

    def build_ai(self):
        # Synthetic training examples: the tree learns pursuit/avoidance decisions.
        X, y = [], []
        for distance in range(2, 31):
            for energy in (0, 1):
                for offset in (-1, 0, 1):
                    X.append([distance, energy, offset])
                    y.append(2 if distance < 5 and energy == 0 else
                             1 if distance < 18 else 0)
        self.ai = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.ai.fit(X, y)

    def build_world(self):
        self.scene = self.render.attachNewNode('arena')
        cube(self.scene, (0,0,-.35), (50,50,.6), (.045,.065,.13,1))
        for p in range(-24,25,4):
            cube(self.scene, (p,0,-.015), (.055,48,.025), (.04,.35,.48,1))
            cube(self.scene, (0,p,-.012), (48,.055,.025), (.04,.35,.48,1))
        for side in (-1,1):
            cube(self.scene, (side*25,0,1.3), (.5,50,2.6), (.04,.26,.4,1))
            cube(self.scene, (0,side*25,1.3), (50,.5,2.6), (.04,.26,.4,1))
        random.seed(15)
        for _ in range(28):
            x,y = random.uniform(-21,21),random.uniform(-21,21)
            if abs(x)<7 and abs(y)<7:
                continue
            h = random.uniform(1.5,5)
            cube(self.scene,(x,y,h/2),(1.2,1.2,h),(.09,.13,.27,1))
            cube(self.scene,(x,y,h+.08),(1.3,1.3,.18),(.06,.75,.95,1))
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(.5,.58,.8,1))
        self.render.setLight(self.render.attachNewNode(ambient))
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(.65,.8,1,1))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(-35,-55,0)
        self.render.setLight(sun_node)
        self.player = self.scene.attachNewNode('player')
        cube(self.player,(0,0,.9),(1.2,1.25,1.5),(.05,.78,.95,1))
        cube(self.player,(0,-.1,1.95),(.85,.85,.8),(.13,.2,.36,1))
        cube(self.player,(0,-.55,2.0),(.65,.12,.22),(.95,.25,.3,1))
        for x in (-.9,.9):
            cube(self.player,(x,0,1.05),(.4,.5,1.15),(.08,.48,.85,1))
            cube(self.player,(x*.42,0,-.08),(.45,.65,.5),(.12,.25,.42,1))
        self.camera.setPos(0,-22,19)
        self.camera.lookAt(0,0,0)

    def build_hud(self):
        self.hud = OnscreenText(text='', pos=(-1.28,.9), align=TextNode.ALeft,
                                scale=.055, fg=(.65,.95,1,1), mayChange=True)
        self.help = OnscreenText(text='WASD MOVER  |  ESPACIO DASH  |  R REINICIAR  |  ESC SALIR',
                                 pos=(0,-.94), scale=.044, fg=(.7,.85,1,1))
        self.message = OnscreenText(text='', pos=(0,.12), scale=.11,
                                    fg=(1,.85,.35,1), mayChange=True)

    def restart_game(self):
        for drone in getattr(self, 'drones', []):
            drone['node'].removeNode()
        for orb in getattr(self, 'orbs', []):
            orb.removeNode()
        if not hasattr(self, 'player'):
            raise RuntimeError('El jugador no fue creado: revisa build_world()')
        self.player.setPos(0,0,0)
        self.health = 100
        self.score = 0
        self.level = 1
        self.elapsed = 0
        self.dash_cd = 0
        self.hit_cd = 0
        self.spawn_timer = 0
        self.finished = False
        self.drones, self.orbs = [], []
        self.session_events = []
        self.message.setText('RECOGE ENERGIA Y ESQUIVA LOS DRONES')
        for _ in range(4):
            self.spawn_orb()
        for _ in range(3):
            self.spawn_drone()

    def spawn_orb(self):
        x,y = random.uniform(-19,19),random.uniform(-19,19)
        orb = cube(self.scene,(x,y,.8),(.8,.8,.8),(.15,1,.52,1))
        self.orbs.append(orb)

    def spawn_drone(self):
        theta = random.uniform(0,2*math.pi)
        x,y = 20*math.cos(theta),20*math.sin(theta)
        node = self.scene.attachNewNode('enemy_drone')
        node.setPos(x,y,0)
        cube(node,(0,0,1.2),(1.7,1.4,.7),(.95,.14,.3,1))
        cube(node,(0,-.75,1.2),(.7,.2,.25),(1,.8,.22,1))
        for sx in (-1,1):
            cube(node,(sx*1.1,0,1.25),(.8,.65,.12),(.35,.25,.8,1))
        self.drones.append({'node':node,'speed':random.uniform(3.5,5.0)+self.level*.4,
                            'energy':random.choice((0,1))})

    def save_statistics(self):
        row = pd.DataFrame([{'fecha':datetime.now().isoformat(timespec='seconds'),
                             'puntuacion':self.score,'nivel':self.level,
                             'segundos':round(self.elapsed,1),'energia_recogida':self.score//10}])
        row.to_csv(CSV_PATH, mode='a', header=not CSV_PATH.exists(), index=False, encoding='utf-8-sig')

    def update_game(self, task):
        dt = min(globalClock.getDt(),.06)
        if self.finished:
            return Task.cont
        self.elapsed += dt
        self.dash_cd = max(0,self.dash_cd-dt)
        self.hit_cd = max(0,self.hit_cd-dt)
        direction = Vec3(float(self.keys['d'])-float(self.keys['a']),
                         float(self.keys['w'])-float(self.keys['s']),0)
        if direction.lengthSquared() > 0:
            direction.normalize()
            speed = 9
            if self.keys['space'] and self.dash_cd == 0:
                speed = 29
                self.dash_cd = 2.3
            self.player.setPos(self.player.getPos()+direction*speed*dt)
            self.player.setH(math.degrees(math.atan2(-direction.x,direction.y)))
        self.player.setX(max(-22,min(22,self.player.getX())))
        self.player.setY(max(-22,min(22,self.player.getY())))
        for orb in self.orbs[:]:
            orb.setH(orb.getH()+90*dt)
            if (orb.getPos()-self.player.getPos()).length() < 1.7:
                self.score += 10
                orb.removeNode()
                self.orbs.remove(orb)
                self.spawn_orb()
                new_level = 1+self.score//50
                if new_level > self.level:
                    self.level = new_level
                    self.spawn_drone()
                    self.message.setText(f'NIVEL {self.level} - MAS DRONES')
        for drone in self.drones:
            node = drone['node']
            delta = self.player.getPos()-node.getPos()
            distance = delta.length()
            if distance > .01:
                decision = int(self.ai.predict([[distance,drone['energy'],0]])[0])
                velocity = drone['speed']*(1.35 if decision == 1 else .8 if decision == 0 else -.4)
                node.setPos(node.getPos()+delta.normalized()*velocity*dt)
                node.lookAt(self.player.getPos())
            if distance < 1.75 and self.hit_cd == 0:
                self.health -= 20
                self.hit_cd = 1.2
                self.message.setText('IMPACTO -20 VIDA')
                if self.health <= 0:
                    self.finished = True
                    self.save_statistics()
                    self.message.setText(f'FIN DEL JUEGO | {self.score} PUNTOS | R REINICIAR')
                    break
        cam_target = self.player.getPos()+Vec3(0,-19,17)
        self.camera.setPos(self.camera.getPos()+(cam_target-self.camera.getPos())*min(1,dt*3))
        self.camera.lookAt(self.player.getPos()+Vec3(0,0,1))
        self.hud.setText(f'NEON ARENA 3D  |  AI BATTLE\nVIDA: {self.health}%\nPUNTOS: {self.score}\nNIVEL: {self.level}\nTIEMPO: {self.elapsed:.0f}s\nDRONES: {len(self.drones)}\nDASH: {"LISTO" if self.dash_cd == 0 else f"{self.dash_cd:.1f}s"}')
        if self.elapsed > 3 and self.hit_cd == 0 and self.message.getText() not in ('',):
            self.message.setText('')
        return Task.cont


if __name__ == '__main__':
    NeonArena().run()
