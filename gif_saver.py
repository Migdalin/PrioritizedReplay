
import imageio
import os
from skimage.transform import resize
import numpy as np

'''
From:  
    https://medium.com/@fabiograetz/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
'''
class GifSaver:
    def __init__(self, memory, agent, save_every_x_episodes=100):
        self.episode_counter = 0
        self.previous_episode_end = 0
        self.save_every_x_episodes = save_every_x_episodes
        self.memory = memory
        self.agent = agent
        self.outputDir = "output"
        os.makedirs(self.outputDir, exist_ok=True)

    def OnEpisodeOver(self):
        self.episode_counter += 1
        if (self.episode_counter >= self.save_every_x_episodes):
            self._CreateGif()
            self.episode_counter = 0
        self.previous_episode_end = self.memory.MaxFrameId

    def _CreateGif(self):
        frames = self._CollectFramesForGif()
        self._GenerateGif(self._GetStartFrameId(), frames)
        
    def _GetStartFrameId(self):
        return self.previous_episode_end + 1
            
    def _CollectFramesForGif(self):
        startFrameId = self._GetStartFrameId()
        maxFrameId = self.memory.MaxFrameId
        framesForGif = []
        for id in range(startFrameId, maxFrameId+1):
            framesForGif.append(self.memory.Frames[id].Contents)
        return framesForGif
            
    def _GenerateGif(self, frame_number, frames_for_gif):
        """
            Args:
                frame_number: Integer, determining the number of the current frame
                frames_for_gif: A sequence of frames of an Atari game (as seen by the network)
                #reward: Integer, Total reward of the episode 
                #path: String, path where gif is saved
        """
        for idx, frame_idx in enumerate(frames_for_gif): 
            frames_for_gif[idx] = resize(frame_idx, (420, 320),
                          mode='constant',
                          preserve_range=True, order=0).astype(np.uint8)
        
        imageio.mimsave(f'{self.outputDir}/{frame_number}.gif', 
                        frames_for_gif, duration=1/15)
        
