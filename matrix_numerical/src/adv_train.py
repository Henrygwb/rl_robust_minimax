import numpy as np
import argparse
from common import env_list, convex_concave, non_convex_non_concave
from env import SubprocVecEnv, MatrixGameEnv, FuncGameEnv
from utils import setup_logger
from ppo_adv import Adv_learn

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# game env.
parser.add_argument("--env", type=int, default=1) #

# random seed.
parser.add_argument("--seed", type=int, default=0)

# number of game environment.
parser.add_argument("--n_games", type=int, default=8) # N_GAME = 8

# The path of the victim policy.
parser.add_argument("--victim_path", type=str, default=None)

# number of steps.
parser.add_argument("--nsteps", type=int, default=2048)

args = parser.parse_args()

# environment selection
GAME_ENV = env_list[args.env]

# victim agent index and model path
VICTIM_PATH = args.victim_path
VICTIM_INDEX = 1

if GAME_ENV == 'Match_Pennies':
    p1_payoffs = np.array([[1, -1], [-1, 1]])
    PAY_OFF = [p1_payoffs, -p1_payoffs]

elif GAME_ENV == 'As_Match_Pennies':
    p1_payoffs = np.array([[2, 0], [-1, 2]])
    PAY_OFF = [p1_payoffs, -p1_payoffs]

elif GAME_ENV == 'CC':
    func = convex_concave
    ACTION_BOUNDARY = 2

elif GAME_ENV == 'NCNC':
    func = non_convex_non_concave
    ACTION_BOUNDARY = 2

else:
    print('Unknow game type.')
    KeyError

# random seed
GAME_SEED = args.seed
# number of game
N_GAME = args.n_games

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
TRAINING_ITER = 20000000 # total training samples.
NSTEPS = 1024  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 2 # number of batches.
NEPOCHS = 4 # number of training iteration in each training iteration.
LR = 3e-4

# Loss function hyperparameters
ENT_COEF = 0.00

LOG_INTERVAL = 1

# SAVE_DIR AND NAME
SAVE_DIR = '../adv-agent-zoo/' + GAME_ENV

EXP_NAME = str(GAME_SEED)


def adv_train(env, logger, out_dir):

    log_callback = lambda logger: env.log_callback(logger)

    def callback(update):
        if update % LOG_INTERVAL == 0:
            log_callback(logger)

    Adv_learn(env_name=GAME_ENV, env=venv, total_timesteps=TRAINING_ITER, lr=LR,
              nsteps=NSTEPS, gamma=GAMMA, nminibatches=NBATCHES, noptepochs=NEPOCHS, 
              ent_coef=ENT_COEF, call_back=callback, out_dir=out_dir, load_path=VICTIM_PATH,
              victim_index=VICTIM_INDEX)


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

        ## self-play training
        adv_train(venv, logger, out_dir)
        venv.close()
