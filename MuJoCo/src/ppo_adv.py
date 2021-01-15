import os
import ray
import pickle
import numpy as np
from env import Adv_Env
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from zoo_utils import LSTM, MLP, remove_prefix, load_model
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


# Custom evaluation during training. This function is called when trainer.train() function ends
def custom_eval_function(trainer, eval_workers):
    """
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    EVAL_NUM_EPISODES = trainer.config['evaluation_num_episodes']
    EVAL_NUM_WOEKER = trainer.config['evaluation_num_workers']
    out_dir = trainer.config['evaluation_config']['out_dir']

    model = trainer.get_policy().get_weights()
    tmp_model = {}
    for k, v in model.items():
        if k == 'default_policy/logstd':
            tmp_model[k] = np.full_like(v, -np.inf)
        else:
            tmp_model[k] = v
    trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_policy().set_weights(tmp_model))

    # Clear up winner stats.
    for w in eval_workers.remote_workers():
        w.foreach_env.remote(lambda env: env.set_winner_info())

    for i in range(int(EVAL_NUM_EPISODES/EVAL_NUM_WOEKER)):
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)

    game_info = []

    # For each worker, get its parallel envs' win information and concate them.
    for w in eval_workers.remote_workers():
        out_info = ray.get(w.foreach_env.remote(lambda env: env.get_winner_info()))
        
        for out in out_info:
            game_info.append(out)

    game_results = np.zeros((3,))
    for game_res in game_info:
        game_results += game_res

    num_games = np.sum(game_results)
    win_0 = game_results[0] * 1.0 / num_games
    win_1 = game_results[1] * 1.0 / num_games
    tie = game_results[2] * 1.0 / num_games

    print('%.2f, %.2f, %.2f' % (win_0, win_1, tie))

    metrics['win_0'] = win_0
    metrics['win_1'] = win_1
    metrics['tie'] = tie

    # write the winning information into txt.
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fid = open(out_dir + '/Log.txt', 'a+')
        fid.write("%d %f %f %f\n" % (trainer.iteration, win_0, win_1, tie))
        fid.close()

    return metrics


def adv_learn(config, nupdates, out_dir):
    # Initialize the ray.
    ray.init()
    trainer = PPOTrainer(env=Adv_Env, config=config)

    for update in range(1, nupdates + 1):
        result = trainer.train()
        # save the model
        m = trainer.get_policy().get_weights()
        m = remove_prefix(m)
        checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i' % update)
        os.makedirs(checkdir)
        savepath = os.path.join(checkdir, 'model')
        pickle.dump(m, open(savepath, 'wb'))
        # save the running mean std of the observations
        obs_filter = trainer.workers.local_worker().get_filters()['default_policy']
        savepath = os.path.join(checkdir, 'obs_rms')
        pickle.dump(obs_filter, open(savepath, 'wb'))
    return 0


def iterative_adv_learn(trainer, nupdates, outer_loop, victim_index, use_rnn, load_pretrain_model, out_dir):

    # You_shall_not_pass:
    # 0, 2, 4 ... : train blocker
    # 1, 3, 5 ... : train runner

    # Initialize the ray.
    ray.init()
    trainer = PPOTrainer(env=Adv_Env, config=config)


    for outer in range(outer_loop):
        # build model to attack
        if outer == 0:
            out_dir = out_dir + '/' + str(0)
        else:
            model_idx = nupdates - 1
            out_dir = out_dir[:-2]
            load_path = out_dir + '/' + str(outer-1) + '/checkpoints/model/' + '%.5d' % model_idx
            out_dir = out_dir + '/' + str(outer)

            # modify the victim_index / model_path in env_config
            config = trainer.config
            config['env_config']['victim_index'] = victim_index

            config['env_config']['model_path'] = load_path
            config['env_config']['init'] = True
            config['evaluation_config'] = { 
                   'out_dir': out_dir,}
            register_env('mujoco_adv_env', lambda config: Adv_Env(config['env_config']))
            if use_rnn:
                ModelCatalog.register_custom_model('custom_rnn', LSTM)
            else:
                ModelCatalog.register_custom_model('custom_mlp', MLP)
            # set up the new trainer
            ray.init()
            trainer = PPOTrainer(env=Adv_Env, config=config)
            if outer >= 2 and load_pretrain_model:
                base_dir = out_dir[:-2]
                # load the trainer
                pretrain_path = base_dir + '/' + str(outer-2) + '/checkpoints/model/' + '%.5d' % model_idx
                pretrain_model = load_model(pretrain_path + '/model')
                pretrain_filter = pickle.load(open(pretrain_path + '/obs_rms', 'rb'))
                trainer.workers.foreach_worker(lambda ev: ev.get_policy().set_weights(pretrain_model))
                trainer.workers.foreach_worker(lambda ev: ev.filters['default_policy'].sync(pretrain_filter))
            
        for update in range(1, nupdates + 1):
            result = trainer.train()
            # save the model
            m = trainer.get_policy().get_weights()
            m = remove_prefix(m)
            checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i' % update)
            os.makedirs(checkdir)
            savepath = os.path.join(checkdir, 'model')
            pickle.dump(m, open(savepath, 'wb'))
            # save the running mean std of the observations
            obs_filter = trainer.workers.local_worker().get_filters()['default_policy']
            savepath = os.path.join(checkdir, 'obs_rms')
            pickle.dump(obs_filter, open(savepath, 'wb'))

        victim_index = 1 - victim_index
        trainer.stop()
        ray.shutdown()

    return 0
