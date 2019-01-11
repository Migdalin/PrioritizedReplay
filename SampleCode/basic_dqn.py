
from collections import deque
import numpy as np
import random
import keras
from keras.layers import Conv2D, Dense, Flatten
from skimage.color import rgb2gray
from skimage.transform import resize
import gym



'''
 Based on agents from rlcode, keon, and probably several others
'''

STATE_DIMENSIONS = (84, 84, 4)


class DqnAgent():
    def __init__(self, action_size):
        self.state_size = STATE_DIMENSIONS
        self.action_size = action_size
        self.batch_state_size = (1, STATE_DIMENSIONS[0], STATE_DIMENSIONS[1], STATE_DIMENSIONS[2])
        
        # parameters about epsilon
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.epsilon_min = 0.1
        self.epsilon_decay_step = 0.0001
                                  
        # parameters about training
        self.batch_size = 32
        self._startTraining = 5000
        self.update_target_rate = 10000
        self.gamma = 0.99
        self.memory = deque(maxlen=400000)

        self.model = self.BuildModel()
        self.target_model = self.BuildModel()
        self.UpdateTargetModel()

    def BuildModel(self):
        result = keras.Sequential()
        result.add(Conv2D(filters=16, kernel_size=8, strides=4, input_shape=self.state_size, activation='relu'))
        result.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        result.add(Flatten())
        result.add(Dense(units=256, activation='relu'))
        result.add(Dense(units=self.action_size))
        result.compile(loss='mse', optimizer='RMSprop')
        return result
    
    # get action from model using epsilon-greedy policy
    def GetAction(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            #state = np.float32(state / 255.0)
            q_value = self.model.predict(state.reshape(self.batch_state_size))
            return np.argmax(q_value[0])

    def GetNoOpAction(self):
        return 0

    def UpdateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def AddToMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def Replay(self, batch_size):
        if(len(self.memory) < self._startTraining):
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            batchState = state.reshape(self.batch_state_size)
            nextBatchState = next_state.reshape(self.batch_state_size)
            
            depreciatedReward = reward
            if not done:
                depreciatedReward += (self.gamma * np.amax(self.target_model.predict(nextBatchState)[0]))
                
            estimatedFutureReward = self.target_model.predict(batchState)
            estimatedFutureReward[0][action] = depreciatedReward
            self.model.fit(batchState, estimatedFutureReward, epochs=1, verbose=0)

class EpisodeManager:
    def __init__(self, agent, environment):
        self._agent = agent
        self._environment = environment
        self._maxStepsPerEpisode = 500
        self._frameBuffer = np.zeros(STATE_DIMENSIONS)
        self._batchSize = 32
        
    def Run(self, numberOfEpisodes):
        for episode in range (numberOfEpisodes):
            score, steps = self.RunOneEpisode()
            self._agent.UpdateTargetModel()
            print(f"Episode: {episode};  Score: {score};  Steps: {steps}")
            
    def RunOneEpisode(self):
        self.OnNextEpisode()
        done = False
        stepCount = 0
        score = 0
        while not done:
            previousState = self._frameBuffer.copy()
            action = self._agent.GetAction(self._frameBuffer)
            _, reward, done, _ = self.NextStep(action)
            score += reward
            self._agent.AddToMemory(previousState, action, reward, self._frameBuffer, done)
            self._agent.Replay(self._batchSize)
            stepCount += 1
            if(stepCount > self._maxStepsPerEpisode):
                break
        return score, stepCount
             
    def OnNextEpisode(self):
        self._environment.reset()
        for _ in range(4):
            self.NextStep(self._agent.GetNoOpAction())

    def NextStep(self, action):
        rawObservation, reward, done, info = self._environment.step(action)
        observation = self.PreProcess(rawObservation)

        for z in range(3):
            self._frameBuffer[:,:,z] = self._frameBuffer[:,:,z+1]
        
        self._frameBuffer[:,:,-1] = observation
        return observation, reward, done, info
            
    '''    
      210*160*3(color) --> 84*84(mono)
      float --> integer (to reduce the size of replay memory)
    '''
    def PreProcess(self, observation):
        processed = np.uint8(
            resize(rgb2gray(observation), (84, 84), mode='constant') * 255)
        return processed
        

class Trainer:
    def Run(self):
        env = gym.make('BreakoutDeterministic-v4')
        agent = DqnAgent(action_size=3)
        mgr = EpisodeManager(agent, env)
        mgr.Run(1000)


trainer = Trainer()
trainer.Run()



        