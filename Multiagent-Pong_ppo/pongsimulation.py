import wrappers
import gym
import torch
import numpy as np
import pongagent
from buffers import Extendedbuffer
from buffers import ExperienceBuffer
from dqn_model import DQN, calc_loss
from torch import optim
from tensorboardX import SummaryWriter
import roboschool.multiplayer
import torch.multiprocessing as mp
from pudb import set_trace
from ppo import *
# TODO: split parameters into simulation parameters and training parameters to pass to run?
class Simulation:
    """
    Simulation for the game of 3D Pong.

    Parameters
    ----------
    params: dict
            Dictionary of all the simulation parameters
    """
    def __init__(self, params, player_n = 0):
        # unpack the parameters:
        #### simulation
        self.device = params["device"]
        self.env_name = params["env_name"]
        self.training_frames = params["training_frames"]
        self.skip_frames = params["skip_frames"]
        self.nactions = params["nactions"]
        self.messages_enabled = params["messages_enabled"]
        self.selfplay = params["selfplay"]
        self.betas = params["betas"]
        self.max_timesteps = params["max_timesteps"]
        self.action_std = params["action_std"]
        self.eps_clip = params["eps_clip"]
        #### qnet model
        self.learning_rate = params["learning_rate"]
        self.sync = params["sync"]
        self.load_from = params["load_from"]
        #### buffer
        self.batch_size = params["batch_size"]
        self.replay_size = params["replay_size"]
        self.nstep = params["nstep"]
        #### agent model
        self.gamma = params["gamma"]
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_decay_rate = params["eps_decay_rate"]
        self.player_n = player_n
        self.double = params["double"]
        # initialize the simulation with shared properties
        self.env = gym.make(self.env_name)   # environment, agent etc. can"t be created jointly in a server simulation
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.net = PPO(self.state_dim, self.action_dim, self.action_std, self.learning_rate, self.betas, self.gamma, self.sync, self.eps_clip)

    def _create_environment(self):
        """
            create a gym environment for the simulation.

            Actions are discretized into nactions and frames are skipped for faster training
            :return: env
            """
        env = gym.make(self.env_name)
        if self.selfplay:
            env.unwrapped.multiplayer(env, game_server_guid="selfplayer", player_n=self.player_n)
        # env = wrappers.action_space_discretizer(env, n=self.nactions)
        env = wrappers.SkipEnv(env, skip=self.skip_frames)
        return env

    def _create_agent(self, env):
        """
            Create agent with buffer for the simulation.

            :return: agent
            """
        # buffer = ExperienceBuffer(self.replay_size)
        buffer = Extendedbuffer(self.replay_size, nstep=self.nstep, gamma=self.gamma)
        agent = pongagent.Pongagent(env, self.player_n, buffer)
        return agent

    def _init_non_shared(self, player_n):
        env = self._create_environment()
        agent = self._create_agent(env)
        writer = SummaryWriter(comment="-"
                                       + "player" + str(player_n)
                                       + "batch" + str(self.batch_size)
                                       + "_eps" + str(self.eps_decay_rate)
                                       + "_skip" + str(self.skip_frames)
                                       + "learning_rate" + str(self.learning_rate))
        return env, agent, writer

    def _fill_buffer(self, agent):
        if self.messages_enabled:
            print("Player populating Buffer ...")
        agent.exp_buffer.fill(agent.env, self.replay_size, self.nstep)
        if self.messages_enabled:
            print("Buffer_populated!")

    def train(self, net, player_n=0):
        self.net = net
        memory = Memory()
        env, agent, writer = self._init_non_shared(player_n)
        self._fill_buffer(agent)
        if self.messages_enabled:
            print("Player %i start training: " %player_n)
        reward_list = []
        time_step = 0

        # training loop
        for i_episode in range(1, self.training_frames+1):
            state = env.reset()
            for t in range(self.max_timesteps):
                time_step +=1
                # Running policy_old:
                action = self.net.select_action(state, memory)
                state, _reward, done, _ = env.step(action)

                # Saving reward and is_terminals:
                memory.rewards.append(_reward)
                memory.is_terminals.append(done)
                reward_list.append(_reward)
                # update if its time
                if time_step % 4000 == 0:
                    self.net.update(memory)
                    memory.clear_memory()
                    time_step = 0
                if done:
                    break



        if self.messages_enabled:
            print("Player %i end training!" %player_n)
        torch.save(net.state_dict(), self.env_name + "end_of_training.dat")

        return np.mean(reward_list[-len(reward_list)//2:])

    # TODO: clean this function!
    def run(self, mode="play"):
        """
        runs the simulation.
        :param mode: str, either "play" or "train"
        :return: mean reward over all episodes with eps_end
        """
        if mode == "train":
            reward = self.train(self.net)
            return reward
        elif mode == "play":
            # Run play.py to see model in action
            pass

        else:
            raise Exception("Mode should be either play or train")


class PongSelfplaySimulation(Simulation):

    def __init__(self, params):
        self.params = params
        super(PongSelfplaySimulation, self).__init__(params)
        game = roboschool.gym_pong.PongSceneMultiplayer()
        self.gameserver = roboschool.multiplayer.SharedMemoryServer(game, "selfplayer", want_test_window=False)
        try:
            mp.set_start_method('spawn')
        except:
            pass
        # self.net.share_memory()

    def run(self, mode="train"):
        # self.net.share_memory()
        sim0 = Simulation(self.params, 0)
        sim1 = Simulation(self.params, 1)
        player_0 = mp.Process(target=sim0.train, args=(self.net, 0,))
        player_1 = mp.Process(target=sim1.train, args=(self.net, 1,))
        player_0.start()
        player_1.start()
        try:
            self.gameserver.serve_forever()
        finally:
            player_0.join()
            player_1.join()


