
from DataModel.sum_tree_node import SumTreeNode

class SingleMemory(SumTreeNode):
    def __init__(self, FirstFrameId, Action, Reward, EpisodeOver):
        self.FirstFrameId = FirstFrameId
        self.Action = Action
        self.Reward = Reward
        self.EpisodeOver = EpisodeOver
