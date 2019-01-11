
import numpy as np
from DataModel.sum_tree import SumTree

"""
This SumTree code is a modified version taken from:
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course

and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
"""
class SumTreeWrapper:
    def __init__(self, capacity):

        self.tree = SumTree(capacity)

        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.  # clipped abs error
        self.dont_divide_by_zero = 0.00001
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority:  looking only at leaf nodes
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        oldData = self.tree.add(max_priority, experience)   # set the max p for new p
        return oldData

    def get_current_data(self):
        return self.tree.get_current_data()
    

    def random_sample(self, batchSize):
        numMemories = np.count_nonzero(self.tree.data)
        indexes = np.random.randint(0, numMemories, batchSize)
        result = self.tree.data[indexes]
        return result

    
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contain the minibatch
        memory_b = []
        
        b_ISWeights = np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        p_min = max(self.dont_divide_by_zero, p_min)  # Until tree is full, np.min(tree) will be zero.
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = data.Priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            #experience = [data]
            
            memory_b.append(data)  #experience)
        
        return memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

