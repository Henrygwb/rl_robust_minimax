import os
import joblib
import argparse
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from common import env_list, convex_concave, as_convex_concave, non_convex_non_concave
from env import SubprocVecEnv, MatrixGameEnv, FuncGameEnv
from utils import setup_logger
from ppo_adv import Adv_learn, iterative_adv_learn

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# game env 0: Match penny, 1: As match penny, 2: Convex-concave function, 3: As-convex-concave function,  4: Non-convex Non-concave function.
parser.add_argument("--env", type=int, default=0) #

# random seed.
parser.add_argument("--seed", type=int, default=0)

# number of game environment.
parser.add_argument("--n_games", type=int, default=8) # N_GAME = 8

# number of steps.
parser.add_argument("--nsteps", type=int, default=2048)

# The index of the victim player.
parser.add_argument("--victim_idx", type=int, default=0)

# The path of the victim policy.
# parser.add_argument("--victim_path", type=str, default='../victim-agent/selfplay/As_CC/player_0/model_0.64030623')
parser.add_argument("--victim_path", type=str, default='../victim-agent/selfplay/Match_Pennies/player_0/model_0')

# The path of the saving the adversarial policy.
parser.add_argument("--save_path", type=str, default='../adv-agent-zoo/Match_Pennies_VictimIDX_0_VictimMODEL_model_0_VictimPARAM_0')

args = parser.parse_args()

# environment selection
GAME_ENV = env_list[args.env]

VICTIM_INDEX = args.victim_idx
VICTIM_PATH = args.victim_path
SAVE_DIR = args.save_path

print(VICTIM_INDEX)
print(VICTIM_PATH)


if GAME_ENV == 'Match_Pennies':
    p1_payoffs = np.array([[1, -1], [-1, 1]])
    PAY_OFF = [p1_payoffs, -p1_payoffs]
    ACTION_BOUNDARY = 1

elif GAME_ENV == 'As_Match_Pennies':
    p1_payoffs = np.array([[2, 0], [-1, 2]])
    PAY_OFF = [p1_payoffs, -p1_payoffs]
    ACTION_BOUNDARY = 1

elif GAME_ENV == 'CC':
    func = convex_concave
    ACTION_BOUNDARY = 2

elif GAME_ENV == 'As_CC':
    func = as_convex_concave
    ACTION_BOUNDARY = 4

elif GAME_ENV == 'NCNC':
    func = non_convex_non_concave
    ACTION_BOUNDARY = 2

else:
    print('Unknow game type.')
    KeyError


# iterative adv traing.
ITERARIVE = True
OUTER_LOOP = 4

if ITERARIVE == True:
    SAVE_DIR = SAVE_DIR +'_iterative_adv_train'
print(SAVE_DIR)

# random seed
GAME_SEED = args.seed
# number of game
N_GAME = args.n_games

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
TRAINING_ITER = 20000000 # total training samples.
# TRAINING_ITER = 100000 # total training samples.
NSTEPS = 1024  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 4 # number of batches.
NEPOCHS = 4 # number of training iteration in each training iteration.
LR = 3e-4

# Loss function hyperparameters
ENT_COEF = 0.00

LOG_INTERVAL = 1

# SAVE_DIR AND NAME

EXP_NAME = str(GAME_SEED)


def adv_train(env, logger, out_dir, victim_index, victim_path, iterative, outer_loop):

    log_callback = lambda logger: env.log_callback(logger)

    def callback(update):
        if update % LOG_INTERVAL == 0:
            log_callback(logger)
    if iterative:
        iterative_adv_learn(env_name=GAME_ENV, env=venv, outer_loop=outer_loop, total_timesteps=TRAINING_ITER, lr=LR,
                            n_steps=NSTEPS, gamma=GAMMA, nminibatches=NBATCHES, noptepochs=NEPOCHS,
                            ent_coef=ENT_COEF, call_back=callback, out_dir=out_dir, load_path=victim_path,
                            victim_index=victim_index, action_boundary=ACTION_BOUNDARY)

    else:
        Adv_learn(env_name=GAME_ENV, env=venv, total_timesteps=TRAINING_ITER, lr=LR,
                  n_steps=NSTEPS, gamma=GAMMA, nminibatches=NBATCHES, noptepochs=NEPOCHS,
                  ent_coef=ENT_COEF, call_back=callback, out_dir=out_dir, load_path=victim_path,
                  victim_index=victim_index, action_boundary=ACTION_BOUNDARY)


if __name__ == "__main__":

        env_name = GAME_ENV
        if args.env < 2:
            single_env = MatrixGameEnv(num_actions=2, payoff=PAY_OFF)
        else:
            single_env = FuncGameEnv(num_actions=2, func=func, env_name=GAME_ENV, action_boundary=ACTION_BOUNDARY)

        # Run multiple envs together.
        venv = SubprocVecEnv([lambda: single_env for i in range(N_GAME)])

        # makedir output
        out_dir, logger = setup_logger(SAVE_DIR, EXP_NAME)

        adv_train(env=venv, logger=logger, out_dir=out_dir, victim_index=VICTIM_INDEX,
                  victim_path=VICTIM_PATH, iterative=ITERARIVE, outer_loop=OUTER_LOOP)

        venv.close()
