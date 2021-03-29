import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import fnmatch, multiprocessing, functools

def find_tfevents(log_dir):
    result = []
    for root, dirs, files in os.walk(log_dir, followlinks=True):
        if root.endswith('rl/tb'):
            for name in files:
                # print(root)
                if fnmatch.fnmatch(name, 'events.out.tfevents.*'):
                    result.append(os.path.join(root, name))
    return result

def read_events_file(events_filename, keys=None):
    events = []
    for event in tf.train.summary_iterator(events_filename):
        for value in event.summary.value:
            if keys is not None and value.tag != keys:
                continue
            events.append(value.simple_value)
    return events

def load_tb_data(log_dir, keys=None):
    event_paths = find_tfevents(log_dir)
    pool = multiprocessing.Pool()
    events_by_path = pool.map(functools.partial(read_events_file, keys=keys), event_paths)
    events_all = []
    for event in events_by_path:
        if len(event) != 0: events_all.append(event)
    return events_all

def load_tb_data_minimax(log_dir, key):
    event_paths = find_tfevents(log_dir)
    event_tmps = []
    for event_path in event_paths:
        for event_tmp in tf.train.summary_iterator(event_path):
            continue
        event_tmps.append(event_tmp)
    # for event_tmp in tf.train.summary_iterator(event_paths[1]):
    #     continue
    values_all = []
    for event_tmp in event_tmps:
        for value in event_tmp.summary.value:
            if key is not None and bool(key.match(value.tag)):
                values_all.append(value.tag)
    pool = multiprocessing.Pool()
    events_all = []
    for value in values_all:
        events_by_path = pool.map(functools.partial(read_events_file, keys=value), event_paths)
        events_all.append(events_by_path)
    events_all = [event for events in events_all for event in events]
    events_final = []
    print('Total numbers of runs %d.' %len(events_all))
    for event in events_all:
        if len(event) == 0: continue
        event_mean = np.mean(event[1000:])
        # if event_mean >= ne-0.5 and event_mean <= ne+0.5:
        events_final.append(event)
    print('Numbers of converged runs %d.' %len(events_final))

    return events_final

def plot_adv_attack(folder, out_dir, exp):

    def save_fig(events, player_n):
        fig, ax = plt.subplots(figsize=(10, 8))
        events = np.array(events)
        # plot min | mean | max
        mean_n = np.mean(events, axis=0)
        min_n = np.min(events, axis=0)
        max_n = np.max(events, axis=0)

        ax.fill_between(x=np.arange(mean_n.shape[0]), y1=min_n, y2=max_n, alpha=0.2, color='r')
        ax.plot(mean_n, linewidth=1, color='r')
        ax.set_xlabel('Iteration.', fontsize=20)
        if player_n == 0:
            ax.set_ylabel('Wining rate of player 0. ', fontsize=20)
        else:
            ax.set_ylabel('Wining rate of player 1. ', fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_xticks([0, int(mean_n.shape[0] / 2), int(mean_n.shape[0])])
        ax.set_yticks([0, 0.5, 1])
        return fig

    exp_folder = folder + '/' + exp
    key_adv = 'win_0'
    key_vic = 'win_1'

    event_adv = load_tb_data(exp_folder, key_adv)
    event_vic = load_tb_data(exp_folder, key_vic)

    fig = save_fig(event_adv, player_n=0)
    fig.savefig(out_dir + '/' + exp + '_party_0.png')
    plt.close()

    fig = save_fig(event_vic, player_n=1)
    fig.savefig(out_dir + '/' + exp + '_party_1.png')
    plt.close()

def plot_minimax(folder, out_dir, exp):

    def save_fig(events, player_n):

        fig, ax = plt.subplots(figsize=(10, 8))
        events = np.array(events)
        # plot min | mean | max
        mean_n = np.mean(events, axis=0)
        min_n = np.min(events, axis=0)
        max_n = np.max(events, axis=0)

        ax.fill_between(x=np.arange(mean_n.shape[0]), y1=min_n, y2=max_n, alpha=0.2, color='r')
        ax.plot(mean_n, linewidth=1, color='r')
        ax.set_xlabel('Iteration.', fontsize=20)
        if player_n == 0:
            ax.set_ylabel('Wining rate of player 0. ', fontsize=20)
        else:
            ax.set_ylabel('Wining rate of player 1. ', fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_xticks([0, int(mean_n.shape[0] / 2), int(mean_n.shape[0])])
        ax.set_yticks([0, 0.5, 1])
        return fig

    exp_folder = folder + '/' + exp
    key_party_0 = re.compile('Win: \d th in 0 player')
    key_party_1 = re.compile('Win: \d th in 1 player')

    event_0 = load_tb_data_minimax(exp_folder, key_party_0)
    event_1 = load_tb_data_minimax(exp_folder, key_party_1)

    fig = save_fig(event_0, player_n=0)
    fig.savefig(out_dir + '/' + exp + '_party_0.png')
    plt.close()

    fig = save_fig(event_1, player_n=1)
    fig.savefig(out_dir + '/' + exp + '_party_1.png')
    plt.close()
    return 0

def plot_minimax_all():
    folder = '/home/xkw5132/wuxian/minimax/minimax_scratch'
    out_dir = '/home/xkw5132/wuxian/minimax/minimax_scratch'
    exp = 'pong_minimax_agents_2_outer_party_id_0_party_0_loop_1_party_1_loop_1_0.0001'
    plot_minimax(folder, out_dir, exp)


def plot_selfplay_all():
    folder = None
    out_dir = None
    exp = None
    plot_selfplay(folder, out_dir, exp)

def plot_adv_all():
    folder = '/home/xkw5132/wuxian/minimax/adv_pretrain'
    out_dir = '/home/xkw5132/wuxian/minimax/adv_pretrain'
    exp = '/pong_adv_0.0003_pretrain_agent_4'
    plot_adv_attack(folder, out_dir, exp)

if __name__ == '__main__':
    plot_adv_all()
