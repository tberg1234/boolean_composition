import numpy as np
import torch
from gym.wrappers import Monitor
import os
import time
from dqn import Agent, DQN, FloatTensor
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame


def video_callable(episode_id):
    return episode_id > 1 and episode_id % 500 == 0


def train(path, env):
    #env = Monitor(env, path, video_callable=video_callable, force=True)
    agent = Agent(env,path=path)
    agent.train()
    return agent


def save(path, agent):
    torch.save(agent.q_func.state_dict(), path)


def load(path, env):
    dqn = DQN(env.action_space.n)
    dqn.load_state_dict(torch.load(path))
    return dqn

start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}

def learn(colour, shape, condition):
    name = colour + shape
    base_path = './models/{}/'.format(name)
    isExist = os.path.exists(base_path)
    if not isExist:
        os.makedirs(base_path)    
        
    env = WarpFrame(CollectEnv(start_positions=None,goal_condition=condition))

    start_pro_time = time.clock()
    print(f"Start training processor time (in seconds): {start_pro_time}")

    agent = train(base_path, env)

    end_pro_time = time.clock()
    print(f"End training processor time (in seconds): {end_pro_time}")
    total_training_time = end_pro_time-start_pro_time
    print(f"Total elapsed training time as processor time (in seconds): {total_training_time}")
    with open(base_path + 'train_time.txt', 'w') as f: f.write(total_training_time)
    save(base_path + 'train_time.txt', end_pro_time - start_pro_time)

    save(base_path + 'model.dqn', agent)

if __name__ == '__main__':

    learn('blue', '', lambda x: x.colour == 'blue')
