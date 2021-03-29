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

# The order of two parties:
# 0: party_0 (model) is in the outer loop -> min_x max_y f(x, y)
# 1: party_1 (opp_model) is in the outer loop  -> max_y min_x f(x, y)
parser.add_argument('--party_order', type=int, default=0)

# Number of training agents in each party.
parser.add_argument("--nagents", type=int, default=2)

# Number of updating loops in outer training each iteration.
parser.add_argument('--update_loop', type=int, default=1)

# Number of inner loops for the inner agent inside the inner loops.
parser.add_argument('--inner_loop', type=int, default=1)

parser.add_argument('--load_pretrain', type=int, default=1)

parser.add_argument('--pretrain_model_path', type=str, default='../victim_agent/from_scratch_max_time_500/selfplay-model')

args = parser.parse_args()


GAME_ENV = 'pong'
# random seed
GAME_SEED = args.seed
# number of game
N_GAME = 1

# server id
SERVER_ID = args.server_id

# Number of agents being trained in each party.
NAGENTS = args.nagents

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
#TRAINING_ITER = 20000000 # total training samples.
TRAINING_ITER = 12000000 # total training samples.
NSTEPS = 4000  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 1 # number of batches.
NEPOCHS = 40 # number of training iteration in each training iteration.
LR = 1e-4

PARTY_ORDER = args.party_order
# pretrain model | path
LOAD_PRETRAIN = args.load_pretrain
PRETRAIN_MODEL_PATH = args.pretrain_model_path

if PARTY_ORDER==0:
    INNER_LOOP_PARTY_0 = args.update_loop
    INNER_LOOP_PARTY_1 = args.update_loop * args.inner_loop
else:
    INNER_LOOP_PARTY_0 = args.update_loop * args.inner_loop
    INNER_LOOP_PARTY_1 = args.update_loop

# Loss function hyperparameters
ENT_COEF = 0.01

LOG_INTERVAL = 1

SAVE_DIR = '../agent-zoo/pong_minimax' + '_agents_' + str(NAGENTS) + '_outer_party_id_' + str(PARTY_ORDER) \
           + '_party_0_loop_' + str(INNER_LOOP_PARTY_0) + '_party_1_loop_' + str(INNER_LOOP_PARTY_1) + '_' + str(LR)


EXP_NAME = str(GAME_SEED)


if __name__ == "__main__":

        # makedir output
        out_dir, logger = setup_logger(SAVE_DIR, EXP_NAME)
        # use to recording the winning rates of agents
        FILE_NAME = out_dir + '/record.txt'

        # create the game server
        game = roboschool.gym_pong.PongSceneMultiplayer()
        gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pong_"+str(SERVER_ID), want_test_window=False,
                                steps=NSTEPS, episode_start=NAGENTS * NAGENTS, file_name=FILE_NAME)

        # player_0 args
        player_0_args = '--total_timesteps={0} --out_dir={1} '\
                        '--n_steps={2} --ent_coef={3} --lr={4} '\
                        '--gamma={5} --nminibatches={6} --noptepochs={7} '\
                        '--nagents={8} --inner_loop_party_0={9} --inner_loop_party_1={10} '\
                        '--player_n={11} --win_info_file={12} --load_pretrain={13} --server_id={14} --pretrain_model_path={15}'.format(TRAINING_ITER, out_dir, NSTEPS, ENT_COEF,
                        LR, GAMMA, NBATCHES, NEPOCHS, NAGENTS, INNER_LOOP_PARTY_0, INNER_LOOP_PARTY_1, 0, FILE_NAME, LOAD_PRETRAIN, SERVER_ID, PRETRAIN_MODEL_PATH)

        player_0_args = player_0_args.split(" ")

        sys_cmd = [sys.executable, 'ppo_minimax.py']
        sys_cmd.extend(player_0_args)
        p0 = subprocess.Popen(sys_cmd)

        # player_1 args
        player_1_args = '--total_timesteps={0} --out_dir={1} '\
                        '--n_steps={2} --ent_coef={3} --lr={4} '\
                        '--gamma={5} --nminibatches={6} --noptepochs={7} '\
                        '--nagents={8} --inner_loop_party_0={9} --inner_loop_party_1={10} '\
                        '--player_n={11} --win_info_file={12} --load_pretrain={13} --server_id={14} --pretrain_model_path={15}'.format(TRAINING_ITER, out_dir, NSTEPS, ENT_COEF,
                        LR, GAMMA, NBATCHES, NEPOCHS, NAGENTS, INNER_LOOP_PARTY_0, INNER_LOOP_PARTY_1, 1, FILE_NAME, LOAD_PRETRAIN, SERVER_ID, PRETRAIN_MODEL_PATH)

        player_1_args = player_1_args.split(" ")
        sys_cmd = [sys.executable, 'ppo_minimax.py']
        sys_cmd.extend(player_1_args)
        p1 = subprocess.Popen(sys_cmd)

        gameserver.serve_forever()
