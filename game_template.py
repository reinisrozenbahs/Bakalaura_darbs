import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()


class Character(object):
    def __init__(self):
        super().__init__()
        self.geometry = []
        self.__angle = 0.0
        self.speed = 0.1
        self.pos = np.zeros((2,))
        self.dir = np.array([0,1])
        self.color = 'r'
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.T = np.identity(3)


    def draw(self):
        x_values = [1,2,3,4]
        y_values = [5,5,5,5]
        for vec2d in self.geometry:
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])
        plt.plot(x_values, y_values)

    def generate_geometry(self):
        pass


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = []


class Player(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])


characters = []
player1 = Player()
characters.append(player1)
characters.append(Asteroid())
characters.append(Asteroid())

is_running = True
def press(event):
    global is_running, player
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

while is_running:
    plt.clf()

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for character in characters: # polymorhism
        character.draw()

    plt.draw()
    plt.pause(1e-2)