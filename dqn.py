import random
import logging
import numpy as np
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt


class DQNAgent(object):

    def __init__(self, toLoad, stage):

        '''
        gamma -> discount factor which controls the importance of future rewards
        alpha -> learning rate / step size

        '''
        self.stage = stage
        self.reward = 0
        self.gamma = 0.9
        #initially alpha = 0.0005
        self.alpha = 0.0001
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.epsilon = 1
        self.actual = []
        self.memory = []
        self.short_memory = np.array([])
        if toLoad == True:
            self.load = True
        else:
            self.load = None

        if self.load==True:
            if self.stage == 'diffusion':
                self.model = self.network("diffusion_model.hdf5")
            elif self.stage == 'osmosis':
                self.model = self.network("osmosis_model.hdf5")
        else:
            self.model = self.network()
        self.agent_target = 1
        self.agent_predict = 0
        #self.epsilon = 0


    
    def get_state(self,game):
        return np.asarray(game.inventory)
    

    def set_reward(self,game):
        self.reward = 0
        if game.fail:
            self.reward = -20
            return self.reward
        if game.wrong_move:
            self.reward = -10
            return self.reward
        if game.correct_move:
            self.reward = 10
            if game.done:
                self.reward = 30
            return self.reward
        
        
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def network(self,weights=None):
        model = Sequential()
        model.add(Dense(32,activation='relu',input_dim=27))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.15))
        #model.add(Dense(120,activation='relu'))
        #model.add(Dropout(0.15))
        #model.add(Dense(120,activation='relu'))
        #model.add(Dropout(0.15))
        model.add(Dense(27,activation='softmax'))
        #decay decay=self.alpha_decay (0.01)
        opt = Adam(self.alpha)
        #loss is mean squared
        model.compile(loss='mse', optimizer=opt,metrics=['accuracy'])
        model.summary
        print(model.summary)

        #set weight to true if loading it in
        if self.load:
            model.load_weights(weights)
        return model

    def replay_new(self,memory):
        if len(memory)>1000:
            minibatch = random.sample(memory,1000)
        else:
            minibatch = memory

        for state,action,reward,next_state,done in minibatch:
            target = reward
            state = np.reshape(state,(1,27))
            next_state = np.reshape(next_state,(1,27))
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][np.argmax(action)] = target
            csv_logger = CSVLogger('training.log', separator=',', append=False)
            self.model.fit(state, target_f,epochs=1,verbose=0,callbacks=[csv_logger])

    def train_short_memory(self,state,action,reward,next_state,done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))
        target_f[0][np.argmax(action)] = target
        self.model.fit(np.array([state]),target_f,epochs=1,verbose=0)

    def plot_graph(self,plt,episodes,rewards):
        loss = rewards.copy()
        plt.plot([i for i in range(episodes)], loss)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.show()

    




