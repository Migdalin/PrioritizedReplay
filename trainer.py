
import os.path
import numpy as np
import gym
import time

from priority_memory import PriorityMemory
from batch_helper import BatchHelper
from dqn_agent import DqnAgent
from dqn_globals import DqnGlobals
from gif_saver import GifSaver

'''
 Based on agents from rlcode, keon, A.L.Ecoffet, and probably several others
'''

class ImagePreProcessor:
    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def downsample(img):
        return img[::2, ::2]
    
    def Preprocess(img):
        shrunk = ImagePreProcessor.downsample(img)
        return ImagePreProcessor.to_grayscale(shrunk)

class EpisodeManager:
    def __init__(self, environment, memory, action_size):
        self._environment = environment
        self._memory = memory
        batchHelper = BatchHelper(memory, DqnGlobals.BATCH_SIZE, action_size)
        self._agent = DqnAgent(action_size, batchHelper)
        self._gifSaver = GifSaver(memory, self._agent)
        
    def ShouldStop(self):
        return os.path.isfile("StopTraining.txt")
        
    def Run(self):
        scoreHistory = []
        while(self.ShouldStop() == False):
            startTime = time.time()
            score, steps = self.RunOneEpisode()
            elapsedTime = time.time() - startTime
            self._agent.OnGameOver(steps)
            self._gifSaver.OnEpisodeOver()
            totalSteps = self._agent.total_step_count
            episode = self._agent.total_episodes
            scoreHistory.append(score)
            print(f"Episode: {episode};  Score: {score};  Steps: {steps}; Time: {elapsedTime:.2f}")
            if(len(scoreHistory) == 10):
                avgScore = np.mean(scoreHistory)
                print(f"Episode: {episode};  Average Score: {avgScore};  Total Steps:  {totalSteps}")
                scoreHistory.clear()
        self._agent.OnExit()

    def OnNextEpisode(self):
        self._environment.reset()
        info = None
        for _ in range(np.random.randint(DqnGlobals.FRAMES_PER_STATE, DqnGlobals.MAX_NOOP)):
            frame, _, done, info = self.NextStep(self._agent.GetFireAction())
            self._memory.AddFrame(frame)
        return info
            
    def NextStep(self, action):
        rawFrame, reward, done, info = self._environment.step(action)
        processedFrame = ImagePreProcessor.Preprocess(rawFrame)
        return processedFrame, reward, done, info
            
    def RunOneEpisode(self):
        info = self.OnNextEpisode()
        done = False
        stepCount = 0
        score = 0
        livesLeft = info['ale.lives']
        while not done:            
            action = self._agent.GetAction()
            frame, reward, done, info = self.NextStep(action)
            score += reward
            if(info['ale.lives'] < livesLeft):
                reward = -1
                livesLeft = info['ale.lives']
            self._memory.AddMemory(frame, action, reward, done)
            self._agent.Replay()
            stepCount += 1
        return score, stepCount
             
class Trainer:
    def Run(self, whichGame):
        env = gym.make(whichGame)
        print(env.unwrapped.get_action_meanings())
        memory = PriorityMemory()
        num_actions = env.action_space.n
        if('Pong' in whichGame):
            num_actions = 4  # Don't need RIGHTFIRE or LEFTFIRE (do we?)
        mgr = EpisodeManager(env, memory, action_size = num_actions)
        mgr.Run()

def Main(whichGame):
    trainer = Trainer()
    trainer.Run(whichGame)

#cProfile.run("Main('PongDeterministic-v4')", "profilingResults.cprof")
#Main('PongDeterministic-v4')
Main('BreakoutDeterministic-v4')



