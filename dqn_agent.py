
import numpy as np
import random
import keras
from pathlib import Path
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Lambda
from keras.optimizers import Adam
import keras.backend as K

from dqn_globals import DqnGlobals


def weighted_is_loss(y_true, y_pred, is_weights):
    hLoss = tf.losses.huber_loss(y_true, y_pred)
    return K.mean(is_weights * hLoss)

'''
 Based on agents from rlcode, keon, A.L.Ecoffet, Thomas Simonini,
 and probably several others
'''

class DqnAgent():
    def __init__(self, action_size, batchHelper):
        self.SetDefaultParameters(action_size, batchHelper)
        self.online_training_model, self.online_predict_model = self.BuildModel()
        _, self.target_predict_model = self.BuildModel()
        #self.LoadModelInfo()
        self.UpdateTargetModel()
    
    def SetDefaultParameters(self, action_size, batchHelper):
        self.state_size = DqnGlobals.STATE_DIMENSIONS
        self.action_size = action_size
        self.BatchHelper = batchHelper
        
        # parameters about epsilon
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.epsilon_min = 0.05
        self.epsilon_decay_step = 0.000002
                                  
        # parameters about training
        self._delayTraining = 20000
        self.update_target_rate = 10000
        self.current_step_count = 0
        self.total_step_count = 0
        self.total_episodes = 0
        self.gamma = 0.99
        self.learning_rate = 0.00001
        
        self.SaveWeightsFilename = "VectorizedDqnWeights.h5"
        self.AgentName = "VectorizedDqnAgent"
        
    def BuildModel(self):
        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(DqnGlobals.STATE_DIMENSIONS, name='frames')
    
        kernelInit = keras.initializers.VarianceScaling(scale=2.0)
    
        conv_1 = Conv2D(filters=32, 
                        kernel_size=8, 
                        strides=4, 
                        input_shape=self.state_size, 
                        activation='relu',
                        kernel_initializer=kernelInit
                        )(frames_input)
        
        conv_2 = Conv2D(filters=64, 
                        kernel_size=4, 
                        strides=2, 
                        activation='relu',
                        kernel_initializer=kernelInit
                        )(conv_1)
        
        conv_3 = Conv2D(filters=64, 
                        kernel_size=3, 
                        strides=1, 
                        activation='relu',
                        kernel_initializer=kernelInit
                        )(conv_2)

        # Flattening the second convolutional layer.
        conv_flattened = Flatten()(conv_3)
        
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        denseHidden = Dense(512, activation='relu', kernel_initializer=kernelInit)(conv_flattened)
        
        advantage = Dense(self.action_size, activation='relu')(denseHidden)
        value = Dense(1)(denseHidden)
        
        policy = Lambda(function = lambda x: x[0]-K.mean(x[0])+x[1], 
                        output_shape = (self.action_size,))([advantage, value])
        
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(self.action_size, kernel_initializer=kernelInit)(policy)
        
        # We multiply the output by the mask!
        actions_input = keras.layers.Input((self.action_size,), name='mask')
        filtered_output = keras.layers.Multiply()([output, actions_input])
        
        #  Prediction
        prediction_model = keras.models.Model(
                inputs=[frames_input, actions_input],
                outputs=filtered_output)
        
        #  Training
        is_weights = keras.layers.Input((1,), name='is_weights')
        y_true = keras.layers.Input((self.action_size,), name='y_true')
        training_model = keras.models.Model(
                inputs=[frames_input, actions_input, is_weights, y_true], 
                outputs=filtered_output)
        training_model.add_loss(weighted_is_loss(y_true, filtered_output, is_weights))
        optimizer = Adam(lr=self.learning_rate)
        training_model.compile(optimizer, loss=None)
        
        return training_model, prediction_model
    
    # get action from model using epsilon-greedy policy
    def GetAction(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
        
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        curState = self.BatchHelper.GetCurrentState()
        q_value = self.online_predict_model.predict([curState, np.ones((1,self.action_size))])
        return np.argmax(q_value[0])

    def GetNoOpAction(self):
        return 0

    def GetFireAction(self):
        return 1

    def LoadModelInfo(self):
        weightsFile = Path(self.SaveWeightsFilename)
        if(weightsFile.is_file()):
            self.online_training_model.load_weights(self.SaveWeightsFilename)
            print("*** Model Weights Loaded ***")

    def SaveModelInfo(self):
        self.target_predict_model.save_weights(self.SaveWeightsFilename)
    
    def UpdateAndSave(self):
        self.UpdateTargetModel()
        self.current_step_count = 0
        self.SaveModelInfo()

    def OnExit(self):
        self.UpdateAndSave()

    def OnGameOver(self, steps):
        self.total_episodes += 1
        self.total_step_count += steps
        self.current_step_count += steps
        if(self.current_step_count >= self.update_target_rate):
            self.UpdateAndSave()

    def UpdateTargetModel(self):
        self.target_predict_model.set_weights(self.online_training_model.get_weights())

    def UpdatePriorities(self, batchInfo, prevQ, nextQ):
        delta = np.abs(nextQ - prevQ)
        self.BatchHelper.UpdateBatchPriorities(batchInfo, delta)
        
    def Replay(self):
        if(self.total_step_count < self._delayTraining):
            return
        
        #start_states, next_states, actions, rewards, gameOvers
        batchInfo = self.BatchHelper.GetBatch()

        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.target_predict_model.predict([batchInfo.nextStates, np.ones(batchInfo.actions.shape)])
        
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[batchInfo.gameOvers] = 0
        
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = batchInfo.rewards + (self.gamma * np.max(next_Q_values, axis=1))
        
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        y_true = batchInfo.actions * Q_values[:,None]
        self.online_training_model.fit(
                x = [batchInfo.startStates, batchInfo.actions, batchInfo.ISWeights, y_true], 
                epochs=1, 
                batch_size=len(batchInfo.startStates), 
                verbose=0
                )

        # Update priorities in memory
        prev_Q_values = self.target_predict_model.predict([batchInfo.startStates, np.ones(batchInfo.actions.shape)])
        prev_Q_values = np.max(prev_Q_values, axis=1)
        self.UpdatePriorities(batchInfo, prev_Q_values, Q_values)
