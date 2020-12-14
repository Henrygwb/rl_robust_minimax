import numpy as np
import argparse
from common import env_list, convex_concave, as_convex_concave, non_convex_non_concave
from env import SubprocVecEnv, MatrixGameEnv, FuncGameEnv
from utils import setup_logger
from ppo_minimax import learn

##################
# Hyper-parameters
##################

parser = argparse.ArgumentParser()
# game env 0: Match penny, 1: As match penny, 2: Convex-concave function, 3: As-convex-concave function,  4: Non-convex Non-concave function.
parser.add_argument("--env", type=int, default=4)

# random seed
parser.add_argument("--seed", type=int, default=0)

# number of game environment.
parser.add_argument("--n_games", type=int, default=8)

# Number of training agents in each party.
parser.add_argument("--nagents", type=int, default=2)

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


# Number of agents being trained in each party.
NAGENTS = args.nagents

# reward discount factor
GAMMA = 0.99

# Training hyperparameters
TRAINING_ITER = 20000000 # total training samples.
# TRAINING_ITER = 400000 # total training samples.
NSTEPS = 1024  # NSTEPS * N_GAME, number of samples in each training update  (TRAINING_ITER/NSTEPS * N_GAME: number of updates)
NBATCHES = 4 # number of batches.
NEPOCHS = 4 # number of training iteration in each training iteration.
LR = 3e-4

# Loss function hyperparameters
ENT_COEF = 0.00

LOG_INTERVAL = 1

# SAVE_DIR AND NAME
SAVE_DIR = '../agent-zoo/minimax/'+ GAME_ENV

EXP_NAME = str(GAME_SEED)


def minimax_train(env, logger, out_dir):

    log_callback = lambda logger: env.log_callback(logger)

    def callback(update):
        if update % LOG_INTERVAL == 0:
            log_callback(logger)

    best_0, best_1 = learn(env_name=GAME_ENV, env=venv, total_timesteps=TRAINING_ITER, out_dir=out_dir, n_steps=NSTEPS,
                           nminibatches=NBATCHES, noptepochs=NEPOCHS, ent_coef=ENT_COEF, lr=LR, gamma=GAMMA,
                           call_back=callback, nagents=NAGENTS, action_boundary=ACTION_BOUNDARY)
    f = open(out_dir+'/best-agents.txt', 'w')
    f.write('The best agent of party 0 in the agent %d.\n' %best_0)
    f.write('The best agent of party 1 in the agent %d.\n' %best_1)
    f.close()


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
        minimax_train(venv, logger, out_dir)
        venv.close()
