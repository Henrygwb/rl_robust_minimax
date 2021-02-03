import os
import ray
import pickle
import timeit
import numpy as np
from env import Adv_Env
from datetime import datetime
from os.path import expanduser
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from zoo_utils import LSTM, MLP, remove_prefix, load_adv_model, add_prefix, load_pretrain_model, create_mean_std
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
        # tmp_model[k] = v
        if k == 'default_policy/logstd':
            tmp_model[k] = np.full_like(v, -np.inf)
        else:
            tmp_model[k] = v
    trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_policy().set_weights(tmp_model))

    # Clear up winner stats.
    for w in eval_workers.remote_workers():
        w.foreach_env.remote(lambda env: env.set_winner_info())

    for i in range(int(EVAL_NUM_EPISODES/EVAL_NUM_WOEKER)):
        # print('%d in %d' %(i, int(EVAL_NUM_EPISODES/EVAL_NUM_WOEKER)))
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


def adv_attacking(config, nupdates, load_pretrained_model, pretrained_model_path, out_dir):

    config['evaluation_config'] = {'out_dir': out_dir}

    ray.init()
    trainer = PPOTrainer(env=Adv_Env, config=config)

    if load_pretrained_model:
        pretrain_model = pickle.load(open(pretrained_model_path + '/model', 'rb'))
        if config['env_config']['env_name'] in ['multicomp/YouShallNotPassHumans-v0', 'multicomp/KickAndDefend-v0']:
            pretrain_model = remove_prefix(pretrain_model)
        pretrain_model = add_prefix(pretrain_model, 'default_policy')
        pretrain_filter = pickle.load(open(pretrained_model_path + '/obs_rms', 'rb'))
        trainer.workers.foreach_worker(lambda ev: ev.get_policy().set_weights(pretrain_model))
        trainer.workers.foreach_worker(lambda ev: ev.filters['default_policy'].sync(pretrain_filter))

    for update in range(1, nupdates + 1):
        start_time = timeit.default_timer()
        result = trainer.train()
        # save the model
        m = trainer.get_policy().get_weights()
        m = remove_prefix(m)
        checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i' % update)
        os.makedirs(checkdir, exist_ok=True)
        savepath = os.path.join(checkdir, 'model')
        pickle.dump(m, open(savepath, 'wb'))

        # save the running mean std of the observations
        obs_filter = trainer.workers.local_worker().get_filters()['default_policy']
        savepath = os.path.join(checkdir, 'obs_rms')
        pickle.dump(obs_filter, open(savepath, 'wb'))

        # Save the running mean std of the rewards.
        for r in range(config['num_workers']):
            remote_worker = trainer.workers.remote_workers()[r]
            rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms))
            rt_rms_tmp = rt_rms_all[0]
            for l in range(len(rt_rms_all)):
                rt_rms_tmp.update_with_other(rt_rms_all[l])

            if r == 0:
                rt_rms = rt_rms_tmp
            else:
                rt_rms.update_with_other(rt_rms_tmp)

        rt_rms = {'rt_rms': rt_rms}
        savepath = os.path.join(checkdir, 'rt_rms')
        pickle.dump(rt_rms, open(savepath, 'wb'))
        print('%d of %d updates, time per updates:' % (update, nupdates))
        print(timeit.default_timer() - start_time)

    folder_time = out_dir.split('/')[-1]
    folder_time = folder_time[0:4] + '-' + folder_time[4:6] + '-' + folder_time[6:8] + '_' + \
                  folder_time[9:11] + '-' + folder_time[11:13]

    # Move log in ray_results to the current output folder.
    default_log_folder = expanduser("~")+'/ray_results'
    log_folders = os.listdir(default_log_folder)
    target_log_folder = [f for f in log_folders if folder_time in f]
    if len(target_log_folder) == 0:
        folder_time = folder_time[:-1] + str(int(folder_time[-1])+1)
        target_log_folder = [f for f in log_folders if folder_time in f]
    for folder in target_log_folder:
        os.system('cp -r '+os.path.join(default_log_folder, folder)+' '+out_dir+'/'+folder)
        os.system('rm -r '+os.path.join(default_log_folder, folder))

    return 0


def iterative_adv_training(config, nupdates, outer_loop, victim_index, use_rnn, load_pretrain_model_it,
                           load_initial, load_pretrained_model_first, pretrained_model_path, pretrained_obs_path, out_dir):

    # Outer_loop:
    # Even iteration [0, 2, 4 ...]: train the original adversarial party (1-victim_index).
    # Odd iteration [1, 3, 5 ...]: train the original victim party (victim_index).
    initial_victim_index = victim_index
    initial_victim_model_path = config['env_config']['victim_model_path']
    for outer in range(outer_loop):
        training_start_time = datetime.now()
        # build model to attack
        out_dir_tmp = out_dir + '/' + str(outer) + '_victim_index_' + str(victim_index)
        config['evaluation_config'] = {'out_dir': out_dir_tmp}

        if outer > 0:
            victim_model_path = out_dir + '/' + str(outer-1) + '_victim_index_' + str((1-victim_index))+ \
                                '/checkpoints/model/' + '%.5d' % (nupdates)

            # modify the victim_party_id / victim_model_path in env_config
            config = trainer.config
            config['env_config']['victim_party_id'] = victim_index

            config['env_config']['victim_model_path'] = victim_model_path

            # Setup everthing
            register_env('mujoco_adv_env', lambda config: Adv_Env(config['env_config']))
            if use_rnn:
                ModelCatalog.register_custom_model('custom_rnn', LSTM)
                config['model']['custom_model'] = 'custom_rnn'
            else:
                ModelCatalog.register_custom_model('custom_mlp', MLP)
                config['model']['custom_model'] = 'custom_mlp'
            config['evaluation_config'] = {'out_dir': out_dir_tmp}

        # set up the new trainer
        ray.init()
        trainer = PPOTrainer(env=Adv_Env, config=config)

        if outer==0 and load_pretrained_model_first:
            pretrain_model, _ = load_pretrain_model(pretrained_model_path, pretrained_model_path)
            if config['env_config']['env_name'] in ['multicomp/YouShallNotPassHumans-v0']:
                pretrain_model['model/logstd'] = pretrain_model['model/logstd'].flatten()
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('default_policy').set_weights(pretrain_model))
            pretrain_filter = create_mean_std(pretrained_obs_path)
            trainer.workers.foreach_worker(lambda ev: ev.filters['default_policy'].sync(pretrain_filter))

        if outer > 0 and load_pretrain_model_it[1-victim_index]:
            if victim_index == initial_victim_index:
                # Adversarial party is the same with the initial adversarial party.
                # load the initial pretrained model as the current adversarial agent.
                pretrain_model, _ = load_pretrain_model(pretrained_model_path, pretrained_model_path)
                if config['env_config']['env_name'] in ['multicomp/YouShallNotPassHumans-v0']:
                    pretrain_model['model/logstd'] = pretrain_model['model/logstd'].flatten()
                pretrain_filter = create_mean_std(pretrained_obs_path)

            else:
                # Adversarial party is the initial victim party.
                # load the initial victim model as the current adversarial agent.
                pretrain_path = initial_victim_model_path
                pretrain_model = pickle.load(open(pretrain_path + '/model', 'rb'))
                if config['env_config']['env_name'] in ['multicomp/YouShallNotPassHumans-v0', 'multicomp/KickAndDefend-v0']:
                    pretrain_model = remove_prefix(pretrain_model)
                pretrain_model = add_prefix(pretrain_model, 'default_policy')
                pretrain_filter = pickle.load(open(pretrain_path + '/obs_rms', 'rb'))

            if outer > 1 and not load_initial[1-victim_index]:
                # Load the latest adversarial model as the current adversarial agent.
                pretrain_path = out_dir + '/' + str(outer-2) + '_victim_index_' + str(victim_index)+ \
                                '/checkpoints/model/' + '%.5d' % (nupdates)
                pretrain_model = load_adv_model(pretrain_path + '/model')
                pretrain_filter = pickle.load(open(pretrain_path + '/obs_rms', 'rb'))

            trainer.workers.foreach_worker(lambda ev: ev.get_policy('default_policy').set_weights(pretrain_model))
            trainer.workers.foreach_worker(lambda ev: ev.filters['default_policy'].sync(pretrain_filter))
            
        for update in range(1, nupdates + 1):
            start_time = timeit.default_timer()
            result = trainer.train()

            # save the model
            m = trainer.get_policy().get_weights()
            m = remove_prefix(m)
            checkdir = os.path.join(out_dir_tmp, 'checkpoints', 'model', '%.5i' % update)
            os.makedirs(checkdir)
            savepath = os.path.join(checkdir, 'model')
            pickle.dump(m, open(savepath, 'wb'))

            # save the running mean std of the observations
            obs_filter = trainer.workers.local_worker().get_filters()['default_policy']
            savepath = os.path.join(checkdir, 'obs_rms')
            pickle.dump(obs_filter, open(savepath, 'wb'))

            # Save the running mean std of the rewards.
            for r in range(config['num_workers']):
                remote_worker = trainer.workers.remote_workers()[r]
                rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms))
                rt_rms_tmp = rt_rms_all[0]
                for l in range(len(rt_rms_all)):
                    rt_rms_tmp.update_with_other(rt_rms_all[l])

                if r == 0:
                    rt_rms = rt_rms_tmp
                else:
                    rt_rms.update_with_other(rt_rms_tmp)

            rt_rms = {'rt_rms': rt_rms}
            savepath = os.path.join(checkdir, 'rt_rms')
            pickle.dump(rt_rms, open(savepath, 'wb'))

            print('%d of %d iterative, victim id: %d, %d of %d updates, time per updates:'
                  % (outer + 1, outer_loop, victim_index, update, nupdates))
            print(timeit.default_timer() - start_time)

        # Move log in ray_results to the current output folder.
        folder_time = training_start_time.strftime('%Y-%m-%d_%H-%M')
        default_log_folder = expanduser("~")+'/ray_results'
        log_folders = os.listdir(default_log_folder)
        target_log_folder = [f for f in log_folders if folder_time in f]

        if len(target_log_folder) == 0:
            folder_time = folder_time[:-1] + str(int(folder_time[-1])+1)
            target_log_folder = [f for f in log_folders if folder_time in f]

        if len(target_log_folder) == 0:
            folder_time = folder_time[:-1] + str(int(folder_time[-1])-1)
            target_log_folder = [f for f in log_folders if folder_time in f]

        for folder in target_log_folder:
            os.system('cp -r '+os.path.join(default_log_folder, folder)+' '+out_dir_tmp+'/'+folder)
            os.system('rm -r '+os.path.join(default_log_folder, folder))

        # Switch victim and adv party id.
        victim_index = 1 - victim_index
        trainer.stop()
        ray.shutdown()

    return 0
