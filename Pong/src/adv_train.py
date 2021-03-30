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
# 0: ucb 1: usenix
parser.add_argument("--attach_type", type=int, default=0)
# iterative round
parser.add_argument("--iterative_round", type=int, default=0)

parser.add_argument("--opp_model_path", type=str, default='../victim_agent/from_scratch_max_time_500/minimax-model')
parser.add_argument("--load_pretrain", type=int, default=0)
# pretrain model path
parser.add_argument("--pretrain_model_path", type=int, default=None)
parser.add_argument("--mimic_model_path", type=str, default=None)

args = parser.parse_args()

GAME_ENV = 'pong'
# random seed
GAME_SEED = args.seed
# number of game
N_GAME = 1

# reward discount factor
GAMMA = 0.99
SERVER_ID = args.server_id

# Training hyperparameters
#TRAINING_ITER = 20000000 # total training samples.
TRAINING_ITER = 5000000 # total training samples.
NSTEPS = 2000  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 20 # number of batches.
NEPOCHS = 4 # number of training iteration in each training iteration.
LR = 3e-4

# Loss function hyperparameters
ENT_COEF = 0.01

LOG_INTERVAL = 1
OPP_MODEL_PATH = args.opp_model_path
MIMIC_MODEL_PATH = args.mimic_model_path
LOAD_PRETRAIN = args.load_pretrain
PRETRAIN_MODEL_PATH = args.pretrain_model_path

if args.attach_type == 0:
   ATTACK_TYPE = 'UCB'
else:
   ATTACK_TYPE = 'USENIX'

ITERATIVE_ROUND = args.iterative_round
SAVE_DIR = '../agent-zoo/pong_adv' + '_' + str(LR) + '_' + ATTACK_TYPE + '_' + 'iterative_round_' + str(ITERATIVE_ROUND)
EXP_NAME = str(GAME_SEED)


if __name__ == "__main__":

        # makedir output
        out_dir, logger = setup_logger(SAVE_DIR, EXP_NAME)
        # use to recording the winning rates of agents
        FILE_NAME = out_dir + '/record.txt'

        # create the game server
        game = roboschool.gym_pong.PongSceneMultiplayer()
        gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pong_adv_"+str(SERVER_ID), want_test_window=False,
                                                               steps=NSTEPS, file_name=FILE_NAME)

        # player_0 args
        player_0_args = '--total_timesteps={0} --out_dir={1} '\
                        '--n_steps={2} --ent_coef={3} --lr={4} '\
                        '--gamma={5} --nminibatches={6} --noptepochs={7} '\
                        '--player_n={8} --win_info_file={9} --adv_party={10} '\
                        '--opp_model_path={11} --mimic_model_path={12} --server_id={13} '\
                        '--load_pretrain={14} --pretrain_model_path={15}'.format(TRAINING_ITER, out_dir, NSTEPS, ENT_COEF,
                        LR, GAMMA, NBATCHES, NEPOCHS,  0, FILE_NAME, 1, OPP_MODEL_PATH, MIMIC_MODEL_PATH, SERVER_ID, LOAD_PRETRAIN, 
                        PRETRAIN_MODEL_PATH)

        player_0_args = player_0_args.split(" ")

        if ATTACK_TYPE == 'UCB':
            sys_cmd = [sys.executable, 'ppo_adv.py']
        else:
            sys_cmd = [sys.executable, 'ppo_usenix_adv.py']


        sys_cmd.extend(player_0_args)
        p0 = subprocess.Popen(sys_cmd)

        # player_1 args
        player_1_args = '--adv_party={0} --player_n={1} --opp_model_path={2} --server_id={3}'.format(0, 1, OPP_MODEL_PATH, SERVER_ID)
        player_1_args = player_1_args.split(" ")

        if ATTACK_TYPE == 'UCB':
           sys_cmd = [sys.executable, 'ppo_adv.py']
        else:
           sys_cmd = [sys.executable, 'ppo_usenix_adv.py']

        sys_cmd.extend(player_1_args)
        p1 = subprocess.Popen(sys_cmd)

        gameserver.serve_forever()
