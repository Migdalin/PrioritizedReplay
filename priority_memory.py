
import numpy as np
from DataModel.single_memory import SingleMemory
from DataModel.single_frame import SingleFrame
from DataModel.sum_tree_wrapper import SumTreeWrapper
from dqn_globals import DqnGlobals

class PriorityMemory:
    def __init__(self):
        # We use a power of two for number of stored memories:  this makes for a 
        # balanced SumTree
        self.MaxActiveMemories = (2**17)  # 131072
        self.MinFrameId = 1  # We remove old frames based on this
        self.MaxFrameId = 0  # We add new frames based on this
        self.Frames = dict()
        self.Memories = SumTreeWrapper(self.MaxActiveMemories)
        self.FramesPerState = DqnGlobals.FRAMES_PER_STATE

    def Forget(self, removedMemory):
        if(removedMemory == None):
            return
        
        while(self.MinFrameId <= removedMemory.FirstFrameId):
            self.Frames.pop(self.MinFrameId)
            self.MinFrameId += 1
        
    def AddMemory(self, newFrameData, action, reward, gameOver):
        newFrameId = self.AddFrame(newFrameData)
        startFrameId = newFrameId - self.FramesPerState
        assert (startFrameId in self.Frames.keys()), "Frame buffer out of sync with memory state."
        newMemory = SingleMemory(int(startFrameId), int(action), int(reward), gameOver)
        oldMemory = self.Memories.store(newMemory)
        self.Forget(oldMemory)

    '''
    Because one memory state can contain multiple frames, we need a way to add frames
    before we start adding new state memories.
    '''
    def AddFrame(self, newFrameData):
        self.MaxFrameId += 1
        theFrame = SingleFrame(self.MaxFrameId, newFrameData)
        self.Frames[theFrame.id] = theFrame
        return theFrame.id

    def Normalize(self, frames):
        normalized = np.float32(frames) / 255.
        return normalized

    def GetBatch(self, batchSize):
        batch_memory = []
        ISWeights = []
        batch_memory, ISWeights = self.Memories.sample(batchSize)
        return batch_memory, ISWeights
    
    def UpdateBatchPriorities(self, treeIndexes, loss):
        self.Memories.batch_update( treeIndexes, loss)

    def GetCurrentState(self):
        curData = self.Memories.get_current_data()
        return self.GetFramesForState(curData.FirstFrameId)
    
    def GetFramesForState(self, stateId):
        tempTuple = ()
        for i in range(self.FramesPerState):
            frameData = self.Frames[stateId+i].Contents
            tempTuple = tempTuple + (frameData.reshape(
                    DqnGlobals.FRAME_DIMENSIONS[0], 
                    DqnGlobals.FRAME_DIMENSIONS[1], 
                    1),)
        
        frameStack = np.concatenate(tempTuple, axis=-1)
        return self.Normalize(frameStack)

