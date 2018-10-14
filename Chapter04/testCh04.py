import gym

e = gym.make("FrozenLake-v0")
e.reset()
print(e.render())