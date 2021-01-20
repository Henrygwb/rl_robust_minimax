import os
import ray
import pickle
import numpy as np
import timeit
from os.path import expanduser
from env import Minimax_Env
from copy import deepcopy
from zoo_utils import remove_prefix, add_prefix
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def policy_mapping_fn(agent_id):
    return agent_id.replace('agent', 'model')


# custom minimax eval function.
def custom_minimax_eval_function(trainer, eval_workers, update, save_idx):

    # The current models in the trainer workers are the updated models for save_idx_*
    # and their latest and best opponent: load_idx_*.

    EVAL_NUM_EPISODES = trainer.config['evaluation_num_episodes']
    EVAL_NUM_WOEKER = trainer.config['evaluation_num_workers']

    num_agents = trainer.config['env_config']['num_agents_per_party']
    out_dir = trainer.config['evaluation_config']['out_dir']

    # Stochastic to deterministic.
    for i in range(num_agents):
        model_idx = 'model_' + str(i)

        tmp_model = trainer.get_policy(model_idx).get_weights()
        tmp_model[model_idx + '/logstd'] = np.full_like(tmp_model[model_idx + '/logstd'], -np.inf)
        eval_workers.foreach_worker(lambda ev: ev.get_policy(model_idx).set_weights(tmp_model))

        filter = trainer.workers.local_worker().get_filters()[model_idx]
        eval_workers.foreach_worker(lambda ev: ev.filters[model_idx].sync(filter))

        opp_model_idx = 'opp_model_' + str(i)

        tmp_opp_model = trainer.get_policy(opp_model_idx).get_weights()
        tmp_opp_model[opp_model_idx + '/logstd'] = np.full_like(tmp_opp_model[opp_model_idx + '/logstd'], -np.inf)
        eval_workers.foreach_worker(lambda ev: ev.get_policy(opp_model_idx).set_weights(tmp_opp_model))

        opp_filter = trainer.workers.local_worker().get_filters()[opp_model_idx]
        eval_workers.foreach_worker(lambda ev: ev.filters[opp_model_idx].sync(opp_filter))

    # Clear up winner stats.
    for w in eval_workers.remote_workers():
        w.foreach_env.remote(lambda env: env.set_winner_info())

    # Check the weights of each eval worker.
    # w_eval_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
    # w_eval_opp_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
    # w_eval_model[0]['model/fully_connected_1/bias']
    # w_eval_model[i]['model/fully_connected_1/bias']
    # w_eval_opp_model[0]['opp_model/fully_connected_1/bias']
    # w_eval_opp_model[i]['opp_model/fully_connected_1/bias']
    # trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_filters())

    for i in range(int(EVAL_NUM_EPISODES / EVAL_NUM_WOEKER)):
        # print("Custom evaluation round", i)
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

    game_results = np.zeros((num_agents, 3))
    for game_res in game_info:
        game_results += game_res

    for i in range(num_agents):
        num_games = np.sum(game_results[i,:])
        win_0 = game_results[i, 0] * 1.0 / num_games
        win_1 = game_results[i, 1] * 1.0 / num_games
        tie = game_results[i, 2] * 1.0 / num_games

        metrics['win_0_%d' %i] = win_0
        metrics['win_1_%d' %i] = win_1
        metrics['tie_%d' %i] = tie

        # write the winning information into txt.
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fid = open(out_dir + '/Log_%s_%d.txt' %(save_idx, i), 'a+')
            if save_idx == 'model':
                fid.write("%d %f %f\n" % (update, win_0, tie))
                print('%s_%d, win %.2f, tie %.2f' % (save_idx, i, win_0, tie))
            else:
                fid.write("%d %f %f\n" % (update, win_1, tie))
                print('%s_%d, win %.2f, tie %.2f' % (save_idx, i, win_1, tie))
            fid.close()

    return metrics


def create_workers(trainer, num_worker):
    # deep copy config
    config = deepcopy(trainer.config)

    # if not evaluation, modify the num_agent to 1
    # if not eval:
    #     config['env_config']['num_agents_per_party'] = 1

    # Collect one trajectory from each worker.
    # How to build per-Sampler (RolloutWorker) batches, which are then
    # usually concat'd to form the train batch. Note that "steps" below can
    # mean different things (either env- or agent-steps) and depends on the
    # `count_steps_by` (multiagent) setting below.
    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    config['batch_mode'] = "complete_episodes"
    config['rollout_fragment_length'] = 1
    config['in_evaluation'] = True

    workers = WorkerSet(
        env_creator=lambda _: Minimax_Env(config['env_config']),
        validate_env=None,
        policy_class=PPOTFPolicy,
        trainer_config=config,
        num_workers=num_worker)
    return workers


def best_opponent(trainer, select_workers, num_agents_per_party, select_num_episodes,
                  select_num_worker, load_idx, save_idx):
    # The current models in party_0 and party_1 are the latest models.
    # For the agents in the current trained party (save_idx), find the best opponents (load_idx)
    # (The opponent that achieve the most returns against the current trained agent.)
    # Return the idx of the best opponents.

    idx = []
    rewards = np.zeros((num_agents_per_party, num_agents_per_party, 2))
    for i in range(num_agents_per_party):
        save_agent = trainer.get_policy(save_idx + '_' + str(i)).get_weights()
        save_agent = remove_prefix(save_agent)

        save_filter = trainer.workers.local_worker().get_filters()[save_idx + '_' + str(i)]

        avg_rewards = []

        for j in range(num_agents_per_party):
            load_agent = trainer.get_policy(load_idx + '_' + str(j)).get_weights()
            load_agent = remove_prefix(load_agent)
            load_filter = trainer.workers.local_worker().get_filters()[load_idx + '_' + str(j)]

            # determine which agent is model/opp_model
            if load_idx == 'model':
                model = load_agent
                opp_model = save_agent
                filter = load_filter
                opp_filter = save_filter
            else:
                model = save_agent
                opp_model = load_agent
                filter = save_filter
                opp_filter = load_filter

            model = add_prefix(model, 'model_%d' %j )
            opp_model = add_prefix(opp_model, 'opp_model_%d' %j)

            model['model_%d/logstd' %j] = np.full_like(model['model_%d/logstd' %j], -np.inf)
            opp_model['opp_model_%d/logstd' %j] = np.full_like(opp_model['opp_model_%d/logstd' %j], -np.inf)

            select_workers.foreach_worker(lambda ev: ev.get_policy('model_%d' %j).set_weights(model))
            select_workers.foreach_worker(lambda ev: ev.get_policy('opp_model_%d' %j).set_weights(opp_model))
            select_workers.foreach_worker(lambda ev: ev.filters['model_%d' %j].sync(filter))
            select_workers.foreach_worker(lambda ev: ev.filters['opp_model_%d' %j].sync(opp_filter))

        # Clear up winner stats.
        for w in select_workers.remote_workers():
            w.foreach_env.remote(lambda env: env.set_winner_info())

        for _ in range(int(select_num_episodes / select_num_worker)):
            ray.get([w.sample.remote() for w in select_workers.remote_workers()])

        # Collect the accumulated episodes on the workers, and then summarize the
        # episode stats into a metrics dict.
        episodes, _ = collect_episodes(
            remote_workers=select_workers.remote_workers(), timeout_seconds=99999)
        # You can compute metrics from the episodes manually, or use the
        # convenient `summarize_episodes()` utility:
        metrics = summarize_episodes(episodes)
        for j in range(num_agents_per_party):
            rewards[i][j][0] = metrics['policy_reward_mean']['model_%d' %j]
            rewards[i][j][1] = metrics['policy_reward_mean']['opp_model_%d' %j]

            if save_idx == 'model':
                avg_rewards.append(metrics['policy_reward_mean']['model_%d' %j])
            else:
                avg_rewards.append(metrics['policy_reward_mean']['opp_model_%d' %j])

        # choose the best opponent according to the rewards
        idx.append(np.argmin(np.array(avg_rewards)))

    return idx, rewards


def minimax_learning(trainer, num_workers, num_agents_per_party, inner_loop_party_0, inner_loop_party_1,
                     select_num_episodes, nupdates, out_dir):

    # MiniMax Training algorithm
    # We define multiple trainable models, one for each party: mapping relation:
    # agent_* -> model_*
    # opp_agent_* -> opp_model_*

    # In each iterations,
    # for each agent_i, we first select its best opponent opp_agent_* of the last iteration,
    # copy its policy to opp_agent_i and update the agent_i with inner_loop_party_0 times. At the last inner loop,
    # we play the updated agent_i with its best opponent and record its winning rate.
    # Save the weights of model_i as the current policy of model_i
    # When done updating for all agent_i, we load the weights of opp_model_* at the last iteration to opp_model_*.

    # for each opp_agent_i, we first select its best opponent agent_* of the last iteration,
    # copy its policy to agent_i and update the opp_agent_i with inner_loop_party_1 times. At the last inner loop,
    # we play the updated opp_agent_i with its best opponent and record its winning rate.
    # Save the weights of opp_model_i as the current policy of opp_model_i
    # When done updating for all opp_agent_i, we load the weights of model_* at the last iteration to model_*.

    def save_policy(trainer, num_agents_per_party, save_idx, update, out_dir, num_workers):

        # save the model / obs_filter
        for i in range(num_agents_per_party):
            m = trainer.get_policy(save_idx + '_' + str(i)).get_weights()
            checkdir = os.path.join(out_dir, 'checkpoints', save_idx + '_' + str(i), '%.5i' % update)
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, 'model')
            pickle.dump(m, open(savepath, 'wb'))

            # Save the running mean std of the observations.
            obs_filter = trainer.workers.local_worker().get_filters()[save_idx + '_' + str(i)]
            savepath = os.path.join(checkdir, 'obs_rms')
            pickle.dump(obs_filter, open(savepath, 'wb'))

            # save the ret_filter
            if trainer.config['env_config']['normalize']:
                for r in range(num_workers):
                    remote_worker = trainer.workers.remote_workers()[r]
                    if save_idx == 'model':
                        rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_0[i]))
                    else:
                        rt_rms_all = ray.get(remote_worker.foreach_env.remote(lambda env: env.ret_rms_1[i]))
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
        return 0

    # Create the workers for evaluation/best opponent selection.
    eval_num_workers = trainer.config['evaluation_num_workers']
    eval_workers = create_workers(trainer, num_worker=eval_num_workers)

    for update in range(1, nupdates + 1):
        start_time = timeit.default_timer()
        for (load_idx, save_idx) in zip(['opp_model', 'model'], ['model', 'opp_model']):
            # print('=================')
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[1][
            #           'model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_1').get_weights())[1][
            #           'model_1/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_0').get_weights())[1][
            #           'opp_model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[1][
            #           'opp_model_1/vffinal/bias'])
            # print('=================')

            # for each save_idx_*, get its best load_idx_*
            idx, _ = best_opponent(trainer, eval_workers, num_agents_per_party, select_num_episodes,
                                   eval_num_workers, load_idx, save_idx)
            # print('=================')
            # print(idx)
            # print('=================')
            tmp_models = []
            tmp_filters = []

            for i in range(num_agents_per_party):
                # Load the best opponents load_idx_* for save_idx_i.
                tmp_model = trainer.get_policy(load_idx + '_' + str(idx[i])).get_weights()
                tmp_model = remove_prefix(tmp_model)
                tmp_model = add_prefix(tmp_model, load_idx + '_' + str(i))
                tmp_models.append(tmp_model)
                tmp_filter = trainer.workers.local_worker().get_filters()[load_idx + '_' + str(idx[i])]
                tmp_filters.append(tmp_filter)

            if save_idx == 'model':
                inner_loop = inner_loop_party_0
            else:
                inner_loop = inner_loop_party_1

            for inner_iter in range(inner_loop):
                # Give load_idx_* to load_idx_i.
                # print('=================')
                for ii in range(num_agents_per_party):
                    trainer.workers.foreach_worker(lambda ev: ev.get_policy(load_idx + '_' + str(ii)).set_weights(tmp_models[ii]))
                    trainer.workers.foreach_worker(lambda ev: ev.filters[load_idx + '_' + str(ii)].sync(tmp_filters[ii]))
                # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[1][
                #           'model_0/vffinal/bias'])
                # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_1').get_weights())[1][
                #           'model_1/vffinal/bias'])
                # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_0').get_weights())[1][
                #           'opp_model_0/vffinal/bias'])
                # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[1][
                #           'opp_model_1/vffinal/bias'])
                _ = trainer.train()

            # After training the agents, run evaluation for current training agents in the save party against their
            # best opponents.
            for ii in range(num_agents_per_party):
                trainer.workers.foreach_worker(
                    lambda ev: ev.get_policy(load_idx + '_' + str(ii)).set_weights(tmp_models[ii]))
                trainer.workers.foreach_worker(lambda ev: ev.filters[load_idx + '_' + str(ii)].sync(tmp_filters[ii]))
            custom_minimax_eval_function(trainer, eval_workers, update, save_idx)
            # print('=================')
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[1][
            #           'model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_1').get_weights())[1][
            #           'model_1/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_0').get_weights())[1][
            #           'opp_model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[1][
            #           'opp_model_1/vffinal/bias'])
            # print('=================')

            # Load the models in the last iteration of the load party.
            if not (update == 1 and load_idx == 'opp_model'):
                if load_idx == 'opp_model':
                    load_update = update - 1
                else:
                    load_update = update
                for i in range(num_agents_per_party):
                    # load the latest model
                    latest_model_path = os.path.join(out_dir, 'checkpoints', load_idx + '_' + str(i), '%.5i' % (load_update),
                                                     'model')
                    tmp_model = pickle.load(open(latest_model_path, 'rb'))
                    trainer.workers.foreach_worker(lambda ev: ev.get_policy(load_idx + '_' + str(i)).set_weights(tmp_model))

                    # load the latest obs_rms
                    latest_norm_path = os.path.join(out_dir, 'checkpoints', load_idx + '_' + str(i), '%.5i' % (load_update),
                                                    'obs_rms')
                    tmp_filter = pickle.load(open(latest_norm_path, 'rb'))
                    trainer.workers.foreach_worker(lambda ev: ev.filters[load_idx + '_' + str(i)].sync(tmp_filter))

            # Save the current models in the save party.
            save_policy(trainer, num_agents_per_party, save_idx, update, out_dir, num_workers)
            # print('=================')
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_0').get_weights())[1][
            #           'model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('model_1').get_weights())[1][
            #           'model_1/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_0').get_weights())[1][
            #           'opp_model_0/vffinal/bias'])
            # print(trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model_1').get_weights())[1][
            #           'opp_model_1/vffinal/bias'])
            # print('=================')
        print('%d of %d updates, time per updates:' % (update, nupdates))
        print(timeit.default_timer() - start_time)

    # return the best agent
    _, rewards = best_opponent(trainer, eval_workers, num_agents_per_party, select_num_episodes,
                               eval_num_workers, 'opp_model', 'model')

    rewards_0 = np.mean(rewards[:, :, 0], axis=1)
    rewards_1 = np.mean(rewards[:, :, 1], axis=0)

    best_0 = np.argmax(rewards_0)
    best_1 = np.argmax(rewards_1)

    f = open(out_dir + '/best-agents.txt', 'w')
    f.write('The best agent of party 0 in the agent %d.\n' % best_0)
    f.write('The best agent of party 1 in the agent %d.\n' % best_1)
    f.close()

    # copy the log
    folder_time = out_dir.split('/')[-1]
    folder_time = folder_time[0:4] + '-' + folder_time[4:6] + '-' + folder_time[6:8] + '_' + \
                  folder_time[9:11] + '-' + folder_time[11:13]

    # Move log in ray_results to the current output folder.
    default_log_folder = expanduser("~") + '/ray_results'
    log_folders = os.listdir(default_log_folder)
    target_log_folder = [f for f in log_folders if folder_time in f]
    if len(target_log_folder) == 0:
        folder_time = folder_time[:-1] + str(int(folder_time[-1]) + 1)
        target_log_folder = [f for f in log_folders if folder_time in f]
    for folder in target_log_folder:
        os.system('cp -r ' + os.path.join(default_log_folder, folder) + ' ' + out_dir + '/' + folder)
        os.system('rm -r ' + os.path.join(default_log_folder, folder))

    return 0
