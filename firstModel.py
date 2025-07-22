#key design questions:
#Q1: what skill should the agent learn?
#Ans: To optimize the price in best way possible for the buyer and seller.
#Q2: what information does the agent need?
#Ans: original price, last offer, reservation price, opponent behaviour, turn number.
#Q3: what actions can the agent take?
#Ans: Accept the price, reject the price, negotiate the price.
#Q4: how do we measure success?
#Ans: mutual agreement(+1 if deal), reward 0 if not.
#Q5: when should episodes end?
#Ans: max turns reached/mutual agreement/offer below threshold

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class NegotiationEnv(gym.Env):
    def __init__(self):
        super(NegotiationEnv, self).__init__()
        self.done = False

        self.action_space=spaces.Discrete(3) # 0: accept, 1: reject, 2: negotiate

        self.min_price=6000
        self.max_price=10000
        self.max_turns=10

        self.observation_space=Box(low=np.array([0.0,0.0,0.0]), high=np.array([1.0,1.0,1.0]), dtype=np.float32)

        self.reset()

      
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn=0
        self.agent_reservation_price=self.min_price
        self.current_offer=self.max_price

        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([self.agent_reservation_price/self.max_price, self.current_offer/self.max_price, self.turn/self.max_turns], dtype=np.float32)
    

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, self.done, False, {}

        reward=0

        if action==0:
            if (self.agent_reservation_price <= self.current_offer ):
                reward=1
            else:
                reward=-1
            self.done=True

        elif action==1:
            self.done=True
            reward=-0.2
        
        elif action==2:
            self.current_offer=np.random.uniform(100,1000)
            self.turn+=1
            if self.turn >= self.max_turns:
                self.done= True

        return self._get_obs(), reward, self.done, False, {}
  
    def render(self):
        print(f"Turn {self.turn} | Offer: {self.current_offer:.2f} | Reservation: {self.agent_reservation_price:.2f}")

    def close(self):
        pass


env = NegotiationEnv()
check_env(env)

model=PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs, _ = env.reset()
done = False

while not done:
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

print("Final Reward:", reward)