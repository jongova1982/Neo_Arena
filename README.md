# Neon Arena 3D — AI Battle

Juego arcade de supervivencia en 3D creado con Python y Panda3D. Controla un
robot, recoge energía y evita drones cuyo movimiento se decide mediante un
árbol de decisión de scikit-learn.

Todo el escenario se genera por código: no se necesitan modelos, texturas,
sonidos ni conexión a Internet.

## Características

- Escenario, personaje, enemigos y objetos creados de forma procedural.
- Drones controlados por un árbol de decisión entrenado al iniciar el juego.
- Sistema de vida, puntuación, niveles y dash con tiempo de recarga.
- Registro local de resultados en un archivo CSV.
- Sin servicios externos ni API de IA.

## Requisitos

- Python 3
- Panda3D
- pandas
- scikit-learn

El proyecto fue probado con Python 3.14.6 y las versiones indicadas en
`requirements.txt`.

## Instalación

Abre una terminal en la carpeta del proyecto. Se recomienda utilizar un
entorno virtual.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Ejecución

En Windows:

```powershell
python neon_arena_3d.py
```

En Linux o macOS:

```bash
python3 neon_arena_3d.py
```

## Controles

| Tecla | Acción |
|---|---|
| `W` | Avanzar |
| `S` | Retroceder |
| `A` | Moverse a la izquierda |
| `D` | Moverse a la derecha |
| `Espacio` | Dash o impulso rápido |
| `R` | Reiniciar la partida |
| `Esc` | Salir |

Si el teclado no responde, haz clic dentro de la ventana del juego para darle
el foco.

## Objetivo

- Recoge los cubos verdes de energía para obtener 10 puntos.
- Evita los drones rojos: cada impacto resta 20 puntos de vida.
- Cada 50 puntos aumenta el nivel y aparece un nuevo dron.
- La partida termina cuando la vida llega a cero.
- Al finalizar se guarda el resultado en `estadisticas_neon_arena.csv`.

El CSV contiene estadísticas locales de las partidas y está excluido del
repositorio mediante `.gitignore`.

## Inteligencia artificial

Al iniciar el juego, scikit-learn entrena en memoria un
`DecisionTreeClassifier` con ejemplos sintéticos. El árbol considera la
distancia al jugador y el estado de energía del dron para escoger entre
distintos comportamientos de aproximación o retirada.

No utiliza aprendizaje por refuerzo ni aprende de las partidas guardadas en el
CSV.

## Estructura del repositorio

```text
.
├── .gitignore
├── neon_arena_3d.py
├── README.md
└── requirements.txt
```

## Solución de problemas

### Falta una dependencia

Instala nuevamente los paquetes con el mismo intérprete que usas para iniciar
el juego:

```powershell
python -m pip install -r requirements.txt
```

### El comando `python` no se reconoce

En Windows prueba:

```powershell
py neon_arena_3d.py
```

### No se puede guardar el CSV

Comprueba que tienes permiso de escritura en la carpeta y que
`estadisticas_neon_arena.csv` no está abierto en otra aplicación.

### La ventana no abre o aparece un error gráfico

Actualiza los controladores de video y comprueba que las dependencias se hayan
instalado correctamente.

## Licencia

Este repositorio todavía no incluye una licencia. Añade un archivo `LICENSE`
antes de publicarlo si deseas permitir que otras personas reutilicen o
modifiquen el código.
