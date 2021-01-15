"""Load two agents for a given environment and perform rollouts, reporting the win-tie-loss."""
import os
import gym
import gym_compete
import os.path as osp
import re
import tempfile
import warnings
import argparse

from wrappers import VideoWrapper
from annotated_gym_compete import AnnotatedGymCompete
from compete import GymCompeteToOurs, game_outcome

from env import env_list

# zoo policy and stable-baseline policy
from video_utils import simulate, load_policy
from ray.rllib.agents.ppo.ppo import PPOTrainer

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=int, default=0)
parser.add_argument("--n_games", type=int, default=1)

parser.add_argument("--config_path", type=str, default="/home/xkw5132/ray_results/PPO_MuJoCo_Env_2021-01-08_21-40-54xu8d11w4/params.pkl")
parser.add_argument("--checkpoint_path", type=str, default="/home/xkw5132/wuxian/model/giles_model")

parser.add_argument("--render", type=bool, default=False)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--timesteps", type=int, default=None)

parser.add_argument("--video", type=bool, default=False)
parser.add_argument("--save_dir", type=str, default=None)

args = parser.parse_args()

# env name
env_name = env_list[args.env]
episodes = args.episodes
timesteps = args.timesteps

config_path = args.config_path
checkpoint_path = args.checkpoint_path


num_env = args.n_games
video = args.video
render = args.render
video_params = None

if video:
   video_params = {
        'save_dir': args.save_dir,        # directory to store videos in.
        'single_file': True,              # if False, stores one file per episode
        'annotated': True,                # for gym_compete, color-codes the agents and adds scores
        'annotation_params': {
            'camera_config': 'default',
            'short_labels': False,
            'resolution': (1280, 720),
            'font': 'times',
            'font_size': 40,
            'draw': True
        },
   }

def get_empirical_score(venv, agents, episodes, timesteps, render):
    """Computes number of wins for each agent and ties. At least one of `episodes`
       and `timesteps` must be specified.

    :param venv: (VecEnv) vector environment
    :param agents: (list<BaseModel>) agents/policies to execute.
    :param episodes: (int or None) maximum number of episodes.
    :param timesteps (int or None) maximum number of timesteps.
    :param render: (bool) whether to render to screen during simulation.
    :return a dictionary mapping from 'winN' to wins for each agent N, and 'ties' for ties."""
    if episodes is None and timesteps is None:
        raise ValueError("At least one of 'max_episodes' and 'max_timesteps' must be non-None.")

    result = {f'win{i}': 0 for i in range(2)}
    result['ties'] = 0

    # This tells sacred about the intermediate computation so it
    # updates the result as the experiment is running
    sim_stream = simulate(venv, agents)

    completed_episodes = 0
    for _, _, done, info in sim_stream:

        if done:
            print('info ', info)
            completed_episodes += 1
            winner = game_outcome(info)
            if winner is None:
               result['ties'] += 1
            else:
               result[f'win{winner}'] += 1

        if episodes is not None and completed_episodes >= episodes:
            break

    return result

def _save_video_or_metadata(env_dir, saved_video_path):
    """
    A helper method to pull the logic for pattern matching certain kinds of video and metadata
    files and storing them as sacred artifacts with clearer names

    :param env_dir: The path to a per-environment folder where videos are stored
    :param saved_video_path: The video file to be reformatted and saved as a sacred artifact
    :return: None
    """
    env_number = env_dir.split("/")[-1]
    video_ptn = re.compile(r'video.(\d*).mp4')
    metadata_ptn = re.compile(r'video.(\d*).meta.json')


def score_agent(env_name, config_path, checkpoint_path, num_env, videos, video_params):

    # create dir for save video 
    #save_dir = video_params['save_dir']
    save_dir = None
    if videos:
        if save_dir is None:
            tmp_dir = tempfile.TemporaryDirectory()
            save_dir = tmp_dir.name
        else:
            tmp_dir = None
        video_dirs = [osp.join(save_dir, str(i)) for i in range(num_env)]

    def env_fn(i):

        env = gym.make(env_name)
        env = GymCompeteToOurs(env)
        if videos:
            if video_params['annotated']:
                if 'multicomp' in env_name:
                    assert num_env == 1, "pretty videos requires num_env=1"
                    env = AnnotatedGymCompete(env, env_name, "zoo", None, "zoo", None, None,
                                              **video_params['annotation_params'])
                else:
                    warnings.warn(f"Annotated videos not supported for environment '{env_name}'")
            env = VideoWrapper(env, video_dirs[i], video_params['single_file'])
        return env

    env = env_fn(0)

    # load agents
    agents = load_policy(env_name, PPOTrainer, config_path, checkpoint_path)
    score = get_empirical_score(env, agents, episodes, timesteps, render)

    env.close()

    if videos:
        for env_video_dir in video_dirs:
            try:
                for file_path in os.listdir(env_video_dir):
                    _save_video_or_metadata(env_video_dir, file_path)

            except FileNotFoundError:
                warnings.warn("Can't find path {}; no videos from that path added as artifacts"
                              .format(env_video_dir))

        if tmp_dir is not None:
            tmp_dir.cleanup()
    print(score)
    return score

def main():
    score_agent(env_name, config_path, checkpoint_path, num_env, video, video_params)

if __name__ == '__main__':
    main()