import gym
from gym import spaces
import numpy as np
from skyjo_gym_game.game.skyjo_game import SkyjoGame as Game
from skyjo_gym_game.game.skyjo_player import SkyjoPlayer
from skyjo_gym_game.game.utils import FIELD_SIZE, TRAIT_MAP, TOTAL_COUNT_VARIANCE
from PIL import Image

class SkyjoEnv(gym.Env):
    
    def __init__(self, num_agents=2):
        super().__init__()
        self.num_agents = num_agents
        self.game = Game(self.num_agents)
        # self.skyjo_game = SkyjoGame(self.num_agents)
        
        # actions = []
        # for _ in range(self.num_agents):
        #     # actions.extend([len(TRAIT_MAP) * 2, 2])
        #     actions.extend([len(TRAIT_MAP), 3])
            
        # self.action_space = spaces.MultiDiscrete(actions)
        self.action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(FIELD_SIZE)])
        # self.action_space = spaces.MultiDiscrete([len(TRAIT_MAP) for _ in range(FIELD_SIZE)])
        
        
        # obs_dict = {str(i): spaces.MultiDiscrete([len(TRAIT_MAP) for _ in range(FIELD_SIZE)]) for i in range(self.num_agents)}
        obs_dict = dict()
        obs_dict['player'] = spaces.MultiDiscrete([len(TRAIT_MAP) for _ in range(FIELD_SIZE)])
        obs_dict['legal_replace'] = spaces.MultiBinary(FIELD_SIZE)
        obs_dict['legal_flip'] = spaces.MultiBinary(FIELD_SIZE)
        
        obs_dict['next_player'] = spaces.MultiDiscrete([len(TRAIT_MAP) for _ in range(FIELD_SIZE)])
        obs_dict['player_amounts'] = spaces.MultiDiscrete([TOTAL_COUNT_VARIANCE, TOTAL_COUNT_VARIANCE])
        obs_dict['other_player_amounts'] = spaces.MultiDiscrete([TOTAL_COUNT_VARIANCE for _ in range(2 * (self.num_agents - 1))])
        obs_dict['discard'] = spaces.Discrete(len(TRAIT_MAP))
        obs_dict['draw'] = spaces.Discrete(len(TRAIT_MAP))
        obs_dict['current_player'] = spaces.Discrete(self.num_agents)
        self.observation_space = spaces.Dict(obs_dict)
        
        
        # self.observation_space = spaces.Box(low=0, high=len(TRAIT_MAP), shape=(5,3,self.num_agents), dtype=np.int32)
        # self.observation_space = spaces.Tuple([
        #     spaces.Box(low=0, high=len(TRAIT_MAP), shape=(4,3, num_agents), dtype=np.int32),
        #     spaces.Discrete(len(TRAIT_MAP)),
        #     spaces.Discrete(len(TRAIT_MAP))
        # ])
        
        
    # def state_builder(self):
    #     self.observation_space['discard'] = 5
    #     print(self.observation_space)
    #     print(self.observation_space['discard'])
    #     print(self.observation_space['1'])
        
        
        
    #     state = []
        
    #     state.append(np.array(self.observation_space[0])[:4, :, self.skyjo_game.current_player()])
    #     state.append(self.observation_space[0][:4, :, self.skyjo_game.next_player()])
    #     state.append(self.observation_space[1])
    #     state.append(self.observation_space[2])
    #     return state
        
        
    def reset(self, seed=None, options=None): # starting state returned
        # self.game.reset()
        del self.game
        self.game = Game(self.num_agents)
        obs = self.game.observe()
        # print('game reset; obs:', obs)
        
        # print(self.action_space)
        # print(self.observation_space)
        # return self.state_builder()
        return obs
        
        
        # pass # return observation
    
    def step(self, action): # preforms provided action and returns next state, reward, done, info
        obs, reward, done, other = self.game.step(action)
        
    
        # return obs, 0, self.game.is_done, None, {} # self.game.observe()
        return obs, reward, done, None, other
        # pass # return observation, reward, done, {}
    

    def render(self, mode='human'): # provides a visual representation of the environment
        # arrays = [player.field for player in self.game.round.players]
        # concat = self.render_four_arrays(arrays)
        # return concat
        pass
    
    def close(self): # shutdown the environment
        pass

    def sample_draw(self, action_size):
        return np.random.randint(0, action_size)
    
    def current_player(self):
        return self.game.current_player()

    def is_done(self):
        return self.game.is_done()
    
    # def render_four_arrays(self, arrays):
        
    #     images = []
    #     for array in arrays:
    #         # Convert the array to a list of characters
    #         characters = [x() for x in array.ravel()]
            
    #         # Convert the list of characters to a string
    #         string = " ".join(characters)
            
    #         # Create a Pillow image from the string
    #         image = Image.fromstring(mode='L', size=array.shape, data=string)
            
    #         # Resize the image to a fixed size
    #         image = image.resize((256, 256))
            
    #         images.append(image)
        
    #     # Concatenate the images horizontally
    #     concatenated = Image.new(mode='RGB', size=(256 * len(images), 256))
    #     x_offset = 0
    #     for image in images:
    #         concatenated.paste(image, (x_offset, 0))
    #         x_offset += image.size[0]
        
    #     return concatenated