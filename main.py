import numpy as np
import matplotlib
import torch

def my_pow(x, y):
    result =1;
    for i in range(y):
        result*=x
    return result

print(my_pow(3,4))

class Animal(object):
    def __init__(self):
        self.hunger_perc = 0.0
    def get_hunger_perc(self):
        return self.hunger_perc
    def eat(self):
        self.hunger_perc-=0.1
        self.hunger_perc = max(0, self.hunger_perc)
    def sleep(self, hours):
        self.hunger_perc += 0.1*hours
        self.hunger_perc = min(1, self.hunger_perc)
    def move(self):
        pass

class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.bones_hidden = 0
    def move(self):
        self.hunger_perc += 0.1
        self.hunger_perc = min(1, self.hunger_perc)
    def bark(self):
        print('Bark!')

class Dog(Animal):
    def __init__(self):
        super().__init__()
        self.bones_hidden = 0
    def move(self):
        self.hunger_perc += 0.01
        self.hunger_perc = min(1, self.hunger_perc)
    def meow(self):
        print('Meow!')

class Robot(object):
    def __init__(self):
        self.battery_perc = 0.0
    def charge(self, hours):
        self.battery_perc += 0.1*hours
        self.battery_perc = min(1, self.battery_perc)
    def move(self):
        self.battery_perc -= 0.1
        self.battery_perc = max(0, self.battery_perc)

