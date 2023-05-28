import numpy as np
import argparse
from utils import setup_logger
import gym
import roboschool
import roboschool.multiplayer
import sys
import subprocess

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# random seed
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--server_id", type=int, default=0)

parser.add_argument('--load_pretrain', type=int, default=0)

parser.add_argument('--pretrain_model_path', type=str, default='../victim_agent/from_scratch_max_time_500/selfplay-model')

args = parser.parse_args()

GAME_ENV = 'pong'
# random seed
GAME_SEED = args.seed
# number of game
N_GAME = 1

# server id
SERVER_ID = args.server_id

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
#TRAINING_ITER = 20000000 # total training samples.
TRAINING_ITER = 8000000 # total training samples.
NSTEPS = 4000  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 1 # number of batches.
NEPOCHS = 1 # number of training iteration in each training iteration.
LR = 1e-4

LOAD_PRETRAIN = args.load_pretrain
PRETRAIN_MODEL_PATH = args.pretrain_model_path

# Loss function hyperparameters
ENT_COEF = 0.01

LOG_INTERVAL = 1

SAVE_DIR = '../agent-zoo/pong_selfplay_a2c' + '_' + str(LR)


EXP_NAME = str(GAME_SEED)


if __name__ == "__main__":

        # makedir output
        out_dir, logger = setup_logger(SAVE_DIR, EXP_NAME)
        # use to recording the winning rates of agents
        FILE_NAME = out_dir + '/record.txt'

        # create the game server
        game = roboschool.gym_pong.PongSceneMultiplayer()
        gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pong_selfplay_"+str(SERVER_ID), want_test_window=False,
                                                               steps=NSTEPS, file_name=FILE_NAME)

        # player_0 args
        player_0_args = '--total_timesteps={0} --out_dir={1} '\
                        '--n_steps={2} --ent_coef={3} --lr={4} '\
                        '--gamma={5} --nminibatches={6} --noptepochs={7} '\
                        '--player_n={8} --win_info_file={9} --load_pretrain={10} --server_id={11} --pretrain_model_path={12}'.format(TRAINING_ITER, out_dir, NSTEPS, ENT_COEF,
                        LR, GAMMA, NBATCHES, NEPOCHS,  0, FILE_NAME, LOAD_PRETRAIN, SERVER_ID, PRETRAIN_MODEL_PATH)

        player_0_args = player_0_args.split(" ")

        sys_cmd = [sys.executable, 'a2c_selfplay.py']
        sys_cmd.extend(player_0_args)
        p0 = subprocess.Popen(sys_cmd)

        # player_1 args
        player_1_args = '--total_timesteps={0} --out_dir={1} '\
                        '--n_steps={2} --ent_coef={3} --lr={4} '\
                        '--gamma={5} --nminibatches={6} --noptepochs={7} '\
                        '--player_n={8} --win_info_file={9} --load_pretrain={10} --server_id={11} --pretrain_model_path={12}'.format(TRAINING_ITER, out_dir, NSTEPS, ENT_COEF,
                        LR, GAMMA, NBATCHES, NEPOCHS, 1, FILE_NAME, LOAD_PRETRAIN, SERVER_ID, PRETRAIN_MODEL_PATH)

        player_1_args = player_1_args.split(" ")
        sys_cmd = [sys.executable, 'a2c_selfplay.py']
        sys_cmd.extend(player_1_args)
        p1 = subprocess.Popen(sys_cmd)

        gameserver.serve_forever()