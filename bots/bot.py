from base import RockBotBase
import numpy as np
from training import get_train


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



def create_action(objects):
    action = {'Version': 1, 'Objects': objects, 'Slot': -1}
    return action


def do_mulligan(scene):
    print(scene)
    mulligan = []
    for card in scene['Self']['Cards']:
        if card['Cost'] > 3:
            mulligan.append(card['RockId'])
    return create_action(mulligan)


def do_play(scene,option):
    print(scene)
    return create_action(option)

def get_options(scene):
    if len(scene['PlayOptions']) == 0:
        return []
    return scene['PlayOptions']

def transform(scene,options):
    ls = []
    for option in options:
        D,H,M,T = option['Damage'],option['Health'],option['Cost'],scene['PermanentResources']
        ls.append(np.array([D,H,M,T]))    
    return np.array(ls)

def do_report(scene):
    print(scene)
    return None


class RockBot(RockBotBase):
    def __init__(self):
        # Build a very simple model
        self.model = Sequential()
        self.model.add(Activation('relu'),input_shape=(4,))
        self.model.add(Dense(30))
        self.model.add(Activation('relu'))
        self.model.add(Dense(30))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('linear'))
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        x,y = get_train()
        self.model.fit(x,y,epochs=5,batch_size=32)
        
        
    def get_mulligan_action(self, scene):
        return do_mulligan(scene)

    def get_play_action(self, scene):
        options = get_options(scene)
        predicts = self.model.predict(transform(options),batch_size=32)        
        return do_play(scene,options[np.argmax(predicts)])

    def report(self, scene):
        return do_report(scene)
    




