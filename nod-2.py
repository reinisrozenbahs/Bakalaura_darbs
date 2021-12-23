import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()


def rotation_mat(degrees):
    theta = np.radians(degrees)
    C = np.cos(theta)
    S = np.sin(theta)
    R = np.array([
        [C, S*(-1), 0],
        [S, C, 0],
        [0, 0, 1]
    ])
    return R

def translation_mat(dx, dy):
    T = np.array([
        [1,0,dx],
        [0,1,dy],
        [0,0, 1]
    ])
    return T

def scale_mat(sx, sy):
    S = np.array([
        [sx,0,0],
        [0,sy,0],
        [0,0,0]
    ])
    return S

def dot(X, Y):
    X=X.tolist()
    Y=Y.tolist()
    X1 = len(X)
    Y1 = len(Y[0])
    temp = 0
    temp_list = []
    return_list = []
    for i in range(X1):
        for j in range(Y1):
            for x_val in X[i]:
                temp += x_val*Y[X[i].index(x_val)][j]
            temp_list.append(temp)
            temp=0
        return_list.append(temp_list)
        temp_list = []
    return np.array(return_list)


def vec2d_to_vec3d(vec2):
    Matrix_I = np.array([[1,0],[0,1],[0,0]])
    vec = np.dot(Matrix_I, vec2)
    vec_z = np.array([[0],[0],[1]])
    return vec+vec_z


def vec3d_to_vec2d(vec3):
    Matrix_I = np.array([[1,0,0],[0,1,0]])
    return np.dot(Matrix_I,vec3)


class Character(object):
    def __init__(self):
        super().__init__()
        self.angle = 0
        self.geometry = []
        self.color = 'g'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.pos = np.zeros((2,))
        self.speed = 0.0

        self.generate_geometry()

    def set_angle(self, angle):
        self.angle = angle # encapsulation
        self.R = rotation_mat(self.angle)

    def get_angle(self):
        return self.angle

    def set_speed(self, val):
        self.speed = val

    def set_pos(self):
        self.pos[0] += self.speed * np.cos(np.radians(self.angle+90))
        self.pos[1] += self.speed * np.sin(np.radians(self.angle+90))

    def draw(self):
        self.T = translation_mat(self.pos[0], self.pos[1])

        x_values = []
        y_values = []

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)
            if isinstance(self, Player):
                vec3d = dot(self.R, vec3d)
            vec3d = dot(self.T, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        self.set_pos()
        plt.plot(x_values, y_values, c=self.color)

        if self.pos[0] > 9:
            if self.get_angle() >0 and self.get_angle() < 90:
                self.set_angle(180-self.get_angle())
            else:
                self.set_angle(180 + (360 - self.get_angle()))
        elif self.pos[0] < -9:
            if self.get_angle() >90 and self.get_angle() < 180:
                self.set_angle(180 - self.get_angle())
            else:
                self.set_angle(180 + 360 - self.get_angle())
        elif self.pos[1] > 9:
                self.set_angle(360-self.get_angle())
        elif self.pos[1] < -9:
            self.set_angle(360-self.get_angle())


class Asteroid(Character):

    def __init__(self):
        super().__init__()
        self.color = 'r'
        self.speed = 0.1
        self.set_pos()
        self.angle = np.random.rand()*360


    def generate_geometry(self):
        circle_list =[]
        x_ofs = np.random.rand() * 20 - 10
        y_ofs = np.random.rand() * 20 - 10
        r_val = np.random.rand() * 1.25
        for i in range(360):
            r_ofs = np.random.rand() / 4
            x = x_ofs + (r_val + r_ofs) * np.cos(np.radians(i))
            y = y_ofs + (r_val + r_ofs) * np.sin(np.radians(i))
            circle_list.append([[x],[y]])
        self.geometry = np.array(circle_list)



class Player(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.S = scale_mat(1,2)
        p_body = np.array([
            [[-1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[-1], [0]]
        ])
        end_list = []
        for vector in p_body:
            vector = vec2d_to_vec3d(vector)
            vector = dot(self.S, vector)
            vector = vec3d_to_vec2d(vector)
            end_list.append(vector)
        self.geometry = np.array(end_list)



characters = []
characters.append(Player())
characters.append(Asteroid())
characters.append(Asteroid())
characters.append(Asteroid())
characters.append(Asteroid())
characters.append(Asteroid())
player = characters[0]

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
    elif event.key == 'up':
        player.set_speed(0.1)
        player.set_pos()
        player.set_speed(0.0)


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

