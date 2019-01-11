
import numpy as np
import collections


BatchInfo = collections.namedtuple('BatchInfo', 
                                   'ISWeights, memories, startStates, nextStates, actions, rewards, gameOvers')

class BatchHelper:
    def __init__(self, memory, batchSize, actionSize):
        self.Memory = memory
        self.BatchSize = batchSize
        self.ActionSize = actionSize

    def GetFrames(self, stateId):
        return self.Memory.GetFramesForState(stateId)

    def GetCurrentState(self):
        singletonBatch = self.Memory.GetCurrentState()
        return np.reshape(singletonBatch, (1,) + singletonBatch.shape)

    def GetBatch(self):
        batchMemory, ISWeights = self.Memory.GetBatch(self.BatchSize)

        startStateList = []
        nextStateList = []
        actionList = []
        rewards = []
        gameOvers = []
        
        for curMemory in batchMemory:
            startStateList.append(self.GetFrames(curMemory.FirstFrameId))
            nextStateList.append(self.GetFrames(curMemory.FirstFrameId+1))
            actionList.append(curMemory.Action)
            rewards.append(curMemory.Reward)
            gameOvers.append(curMemory.EpisodeOver)
            
        startStates = np.stack(startStateList, axis=0)
        nextStates = np.stack(nextStateList, axis=0)
        actions = np.eye(self.ActionSize)[np.array(actionList)]
        
        result = BatchInfo(
                ISWeights = ISWeights,
                memories = batchMemory,
                startStates = startStates,
                nextStates = nextStates,
                actions = np.array(actions), 
                rewards = np.array(rewards), 
                gameOvers = np.array(gameOvers)
                )
        
        return result

    def UpdateBatchPriorities(self, batchInfo, loss):
        self.Memory.UpdateBatchPriorities([m.TreeIndex for m in batchInfo.memories], loss)
