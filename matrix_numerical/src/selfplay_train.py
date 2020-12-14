import numpy as np
import argparse
from common import env_list, convex_concave, as_convex_concave, non_convex_non_concave
from env import SubprocVecEnv, MatrixGameEnv, FuncGameEnv
from utils import setup_logger
from ppo_selfplay import learn

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# game env 0: Match penny, 1: As match penny, 2: Convex-concave function, 3: As-convex-concave function,  4: Non-convex Non-concave function.
parser.add_argument("--env", type=int, default=3)

# random seed
parser.add_argument("--seed", type=int, default=0)

# number of game environment.
parser.add_argument("--n_games", type=int, default=8)

# The model used as the opponent. latest, random, best
parser.add_argument("--opp_model", type=str, default='latest')

args = parser.parse_args()

GAME_ENV = env_list[args.env]

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

# training agent id
TRAIN_ID = 1

if args.opp_model == 'latest':
    OPP_MODEL = 0

elif args.opp_model == 'random':
    OPP_MODEL = 1

elif args.opp_model == 'best':
    OPP_MODEL = 2

else:
    print('unknown option of which model to be used as the opponent model, default as the latest model.')
    OPP_MODEL = 0

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
TRAINING_ITER = 20000000 # total training samples.
# TRAINING_ITER = 400000 # total training samples.
NSTEPS = 1024  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 2 # number of batches.
NEPOCHS = 4 # number of training iteration in each training iteration.
LR = 3e-4

# Loss function hyperparameters
ENT_COEF = 0.00

LOG_INTERVAL = 1

# SAVE_DIR AND NAME
SAVE_DIR = '../agent-zoo-test/'+ GAME_ENV + '_PLAYER_' + str(TRAIN_ID) + '_OPPO_Model_' + str(OPP_MODEL)

EXP_NAME = str(GAME_SEED)


def selfplay_train(env, logger, out_dir):

    log_callback = lambda logger: env.log_callback(logger)

    def callback(update):
        if update % LOG_INTERVAL == 0:
            log_callback(logger)

    learn(env_name=GAME_ENV, env=venv, opp_method=OPP_MODEL, total_timesteps=TRAINING_ITER, n_steps=NSTEPS,
          nminibatches=NBATCHES, noptepochs=NEPOCHS, ent_coef=ENT_COEF, lr=LR, gamma=GAMMA, call_back=callback,
          out_dir=out_dir, train_id=TRAIN_ID, action_boundary=ACTION_BOUNDARY)


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
        selfplay_train(venv, logger, out_dir)
        venv.close()
