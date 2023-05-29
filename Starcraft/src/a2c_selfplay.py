import os
import ray
import timeit
import pickle
import random
import numpy as np
from zoo_utils import add_prefix, remove_prefix
from copy import deepcopy
from env import Starcraft_Env
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


# Custom evaluation during training. This function is called when trainer.train() function ends
def custom_symmtric_eval_function(trainer, eval_workers):
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

    model = trainer.get_policy('model').get_weights()
    opp_model = trainer.get_policy('opp_model').get_weights()

    tmp_model = {}
    tmp_opp_model = {}
    # Eval_workers should load latest model

    # In even iteration, use model as the current policy.
    # In odd iteration, use opp_model as the current policy.
    # Copy the current policy to eval_workers' weight

    if trainer.iteration % 2 == 0:
        for (k1, v1), (k2, _) in zip(model.items(), opp_model.items()):
            tmp_model[k1] = v1
            tmp_opp_model[k2] = v1
    else:
        for (k1, _), (k2, v2) in zip(model.items(), opp_model.items()):
            tmp_model[k1] = v2
            tmp_opp_model[k2] = v2
        
    trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_policy('model').set_weights(tmp_model))
    trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_policy('opp_model').set_weights(tmp_opp_model))

    # Clear up winner stats.
    for w in eval_workers.remote_workers():
        w.foreach_env.remote(lambda env: env.set_winner_info())

    # Check the weights of each eval worker.
    # w_eval_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
    # w_eval_opp_model = eval_workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
    # local_worker_model: w_eval_model[0]['model/fully_connected_1/bias']
    # remote_eval_i_worker_model: w_eval_model[i]['model/fully_connected_1/bias']
    # local_worker_opp_model: w_eval_opp_model[0]['opp_model/fully_connected_1/bias']
    # remote_eval_i_worker_opp_model: w_eval_opp_model[i]['opp_model/fully_connected_1/bias']
    # If using model/opp_model as the current policy,
    # all remote workers should have the same parameters with model/opp_model.
    # All filters: trainer.evaluation_workers.foreach_worker(lambda ev: ev.get_filters())

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
        env_creator=lambda _: Starcraft_Env(config['env_config']),
        #validate_env=None,
        policy_class=A3CTFPolicy,
        trainer_config=config,
        num_workers=num_worker)
    return workers


def symmtric_best_opponent(trainer, nupdates, select_workers, select_num_episodes, \
                  select_num_worker, load_idx, save_idx):
    out_dir = trainer.config['evaluation_config']['out_dir']
    
    # Load model/filter of save_idx agent
    save_agent = trainer.get_policy(save_idx).get_weights()
    select_workers.foreach_worker(lambda ev: ev.get_policy(save_idx).set_weights(save_agent))

    # Return the rewards
    # To accelrate the computation, we compute at most 10 agents randomly sampled
    if nupdates > 10:
        possible_updates = np.random.choice(np.arange(1, nupdates-1), 9).tolist()
        possible_updates.append(nupdates - 1)
    else:
        possible_updates = np.arange(1, nupdates).tolist()
    avg_rewards = []

    for update in possible_updates:
        model_path = os.path.join(out_dir, 'checkpoints', 'model', '%.5i'% update, 'model')
        tmp_model = pickle.load(open(model_path, 'rb'))
        tmp_model = add_prefix(tmp_model, load_idx)
        select_workers.foreach_worker(lambda ev: ev.get_policy(load_idx).set_weights(tmp_model))

        # Clear up winner stats.
        for w in select_workers.remote_workers():
            w.foreach_env.remote(lambda env: env.set_winner_info())

        for _ in range(int(select_num_episodes / select_num_worker)):
            ray.get([w.sample.remote() for w in select_workers.remote_workers()])

        # Collect the accumulated episodes on the workers, and then summarize the
        # episode stats into a metrics dict.
        episodes, _ = collect_episodes(
            remote_workers=select_workers.remote_workers(), timeout_seconds=99999)
        metrics = summarize_episodes(episodes)
        avg_rewards.append(metrics['policy_reward_mean'][save_idx])
    idx = np.argmin(np.array(avg_rewards))
    return possible_updates[idx]


# Define the policy_mapping_function.
# 'agent_0'  ---->   model
# 'agent_1'  ---->   opp_model
def policy_mapping_fn(agent_id):
    # get agent id via env.reset()
    ret = 'model' if agent_id == 'agent_0' else 'opp_model'
    return ret


def symmtric_learning(trainer, num_workers, nupdates, opp_method, select_num_episodes, out_dir):
    # Symmtric Training algorithm
    # We define two trainable models, one for each party: mapping relation: agent_0 -> model  agent_1 -> opp_model.

    # Split the total training iteration into two halves.
    # In the even iterations, we load a previous (random/latest) model for agent_1 and train agent 0, save the model
    # of agent 0 as the trained policy after this iteration.
    # Copy the weights of model to the opp_model.
    # In the odd iterations, we load a previous (random/latest) model for agent_0 and train agent 1, save the opp_model
    # of agent 1 as the trained policy after this iteration.
    # Copy the weights of opp_model to the model.

    # In the even iterations, update model, sample a previous policy for opp_model.
    # In the odd iterations, update opp_model, sample a previous policy for model.

    eval_num_workers = trainer.config['evaluation_num_workers']
    eval_workers = create_workers(trainer, num_worker=eval_num_workers)

    for update in range(1, nupdates + 1):
        start_time = timeit.default_timer()
        if update == 1:
            print('Use the initial agent as the opponent.')
        else:
            if opp_method == 0:
                print('Select the latest model')
                selected_opp_model = update - 1
            elif opp_method == 1:
                print('Select the random model')
                selected_opp_model = round(np.random.uniform(1, update - 1))
            else:
                print('Select the best model')
                if update % 2 == 0:
                    load_idx = 'opp_model'
                    save_idx = 'model'
                else:
                    load_idx = 'model'
                    save_idx = 'opp_model'
                selected_start_time = timeit.default_timer()
                selected_opp_model = symmtric_best_opponent(trainer, update, eval_workers, select_num_episodes, \
                                                   eval_num_workers, load_idx, save_idx)
                # print(timeit.default_timer() - selected_start_time)
                # print(selected_opp_model)

            # In the even iteration, sample a previous policy for opp_model.
            # In the odd iteration, sample a previous policy for model.
            model_path = os.path.join(out_dir, 'checkpoints', 'model', '%.5i'%selected_opp_model, 'model')
            tmp_model = pickle.load(open(model_path, 'rb'))
            if update % 2 == 0:
                prefix = 'opp_model'
            else:
                prefix = 'model'
            tmp_model = add_prefix(tmp_model, prefix)
            trainer.workers.foreach_worker(lambda ev: ev.get_policy(prefix).set_weights(tmp_model))


        # Update both model and opp_model.
        result = trainer.train()
        # Forward pass (run function): each worker at each time: call sample function in rollout_worker.py (line 615).
        # It then call SyncSampler in sampler.py (line 118), which will then call_env_runner in sampler.py (line 412).
        # This function will get one step observation for each agent in each env and compute actions, rnn_state, action
        # distribution stats, and predicted value function. For each agent in each env, the policy network takes as
        # input an observation [1, 1, observation_space] (RNN) and output an action [1, action_space].
        # The outputs have shape [nenv_per_worker, object_dimension], e.g., #action [nenv_per_worker, action_shape].
        # call_env_runner function will be call ROLLOUT_FRAGMENT_LENGTH times. The data will be returned to sample
        # function in rollout_worker.py. The sample function will first get a data dict batch, which contains the
        # collected data for each agent in the current worker.
        # Each item has the shape [nenv_per_worker*ROLLOUT_FRAGMENT_LENGTH, object_dimension].

        # Backward training pass (train function): call TrainTFMultiGPU in train_ops.py (line 84).
        # It first takes as input shape of each item [nenv_per_worker*ROLLOUT_FRAGMENT_LENGTH, object_dimension].
        # It then transform samples into feed_dict (named as tuples),
        # The keys of sumoants with rnn model are:
        # Tensor("opp_model/obs:0", shape=(?, 137), dtype=float32)
        # Tensor("opp_model/action:0", shape=(?, 8), dtype=float32)
        # Tensor("opp_model/Placeholder:0", shape=(?, 128), dtype=float32)
        # Tensor("opp_model/Placeholder_1:0", shape=(?, 128), dtype=float32)
        # Tensor("opp_model/Placeholder_2:0", shape=(?, 128), dtype=float32)
        # Tensor("opp_model/Placeholder_3:0", shape=(?, 128), dtype=float32)
        # Tensor("opp_model/action_logp:0", shape=(?,), dtype=float32)
        # Tensor("opp_model/action_dist_inputs:0", shape=(?, 16), dtype=float32)
        # Tensor("opp_model/vf_preds:0", shape=(?,), dtype=float32)
        # Tensor("opp_model/value_targets:0", shape=(?,), dtype=float32)
        # Tensor("opp_model/advantages:0", shape=(?,), dtype=float32)
        # Tensor("opp_model/seq_lens:0", shape=(?,), dtype=int32)

        # After prepare the feed_dict, it then updates the models.
        # for num_sgd_iter: for (train_batch_size/sgd_minibatch_size): update with the current sgd_minibatch data.

        # Convert the given input shapes to the ones required by the rnn models.
        # Code: dynamic_tf_policy.py line 283 -> modelv2.py line 209 -> recurrent_net.py line 66
        # (rnn_sequencing.py line 116)
        # Example: Suppose input length (sgd_minibatch_size) is 5 [A, B, B, C, C] (observation),
        # seq_len = [1, 2, 2]
        # where A, B, C represents the trajectories collected from different envs in different workers.
        # First padding env to max trajectory length pad_obs = [A, *, B, B, C, C].
        # Then compute the max_seq_len = pad_obs.shape[0]//seq_len.shape[0], internal_batch_size = seq_len.shape[0]
        # Reshape observation input as [[A, *], [B, B], [C, C]] [internal_batch_size, max_seq_len, dim]
        # Each state with shape [internal_batch_size, dim]
        # seq_len [internal_batch_size,] [1, 2, 2]. each element is the the length of the corresponding batch, it will
        # be given to tf.sequence_mask(seq_len[0]) -> [True], which will mask out the padded step in this sequence.

        # Copy the weights of local workers to remote workers.
        # Return the collected samples and loss values.

        # nenv_per_worker 2 * ROLLOUT_FRAGMENT_LENGTH 100
        # Training batch size. TRAIN_BATCH_SIZE = 200
        # Minibatch size. Num_epoch = train_batch_size/sgd_minibatch_size. TRAIN_MINIBATCH_SIZE = 100

        # Ray will implicitly call custom_eval_function.

        # Sync model parameters.
        # In the even iteration, save model as the current policy and copy the weights of model to opp_model.
        # In the odd iteration, save opp_model as the current policy and copy the weights of opp_model to model.

        model = trainer.get_policy('model').get_weights()
        opp_model = trainer.get_policy('opp_model').get_weights()
        tmp_model = {}

        if update % 2 == 0:
            for (k1, v1), (k2, _) in zip(model.items(), opp_model.items()):
                tmp_model[k2] = v1
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').set_weights(tmp_model))
        else:
            for (k1, _), (k2, v2) in zip(model.items(), opp_model.items()):
                tmp_model[k1] = v2
            trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').set_weights(tmp_model))
        # Check model parameters.
        # ww = trainer.workers.foreach_worker(lambda ev: ev.get_policy('model').get_weights())
        # ww_opp = trainer.workers.foreach_worker(lambda ev: ev.get_policy('opp_model').get_weights())
        # Check the length of ww/ww_opp. The first one is local worker, the others are remote works.

        # After sync the model weights, save the current policy and rms parameters.
        m = trainer.get_policy('model').get_weights()
        m = remove_prefix(m)
        checkdir = os.path.join(out_dir, 'checkpoints', 'model', '%.5i' % update)
        os.makedirs(checkdir, exist_ok=True)
        savepath = os.path.join(checkdir, 'model')
        pickle.dump(m, open(savepath, 'wb'))

        print('%d of %d updates, time per updates:' %(update, nupdates))
        print(timeit.default_timer() - start_time)

    return 0