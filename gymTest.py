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
import numpy as np

class NegotiationEnv(gym.Env):
    def __init__(self):
        super(NegotiationEnv, self).__init__()
        self.done = False

        self.action_space=spaces.Discrete(3) # 0: accept, 1: reject, 2: negotiate

        self.min_price=6000
        self.max_price=10000
        self.max_turns=10

        self.observation_space=spaces.Dict({
            "reservation_price": spaces.Box(low=self.min_price, high=self.max_price, shape=(1,), dtype=np.float32),
            "last_offer": spaces.Box(low=self.min_price, high=self.max_price, shape=(1,), dtype=np.float32),
            "turn": spaces.Discrete(self.max_turns+1),
        })

        self.reset()

      
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn=0
        self.agent_reservation_price=np.random.uniform(200,800)
        self.current_offer=np.random.uniform(100,1000)

        return self._get_obs(), {}
    
    def _get_obs(self):
        return {
            "reservation_price": np.array([self.agent_reservation_price], dtype=np.float32),
            "last_offer": np.array([self.current_offer], dtype=np.float32),
            "turn": self.turn
        }
    

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        reward=0

        if action==0:
            if (self.agent_reservation_price <= self.current_offer ):
                reward=1
            else:
                reward=-1
            self.done=True

        elif action==1:
            self.done=True
            reward=0
        
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

obs, info = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Replace with your agent’s policy later
    obs, reward, done, truncated, info = env.step(action)

print("Final Reward:", reward)
env.close()
