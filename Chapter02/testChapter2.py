import gym
e = gym.make('CartPole-v0')
obs = e.reset()
print(obs);
obs = e.step(e.action_space.sample())
print(obs)