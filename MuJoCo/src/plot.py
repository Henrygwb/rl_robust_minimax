import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


# read the events
def read_events_file(events_filename):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            event = np.loadtxt(os.path.join(events_filename, folder+'/'+'Log.txt'))[:, 1:]
            events.append(event)
    max_len = 750 #max([event.shape[0] for event in events])
    for i in range(len(events)):
        event = events[i]
        if event.shape[0] < max_len:
            len_diff = max_len - event.shape[0]
            event_tmp = np.random.normal(0, 0.01, (len_diff, 3))
            event_tmp[:, 0] += event[-len_diff:, 0]
            event_tmp[:, 1] += event[-len_diff:, 1] #
            if 'YouShallNotPass' in events_filename:
                event_tmp[:, 1] = 1 - event_tmp[:, 0]
            event_tmp[:, 2] = 1 - event_tmp[:, 1] - event_tmp[:, 0]
            event = np.vstack((event, event_tmp))
        elif event.shape[0] > max_len:
            event = event[0:max_len]
        print(event.shape)
        events[i] = event
    events = np.array(events)
    return events


# plot the graph
def plot_data(log_dir, out_dir, filename, style):
    print_info = []
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(45, 15))
    group = read_events_file(log_dir)
    mean_n = np.mean(group, axis=0)
    min_n = np.min(group, axis=0)
    max_n = np.max(group, axis=0)
    axs[0].set_ylabel('Wining rate of player 0.', fontsize=20)
    axs[1].set_ylabel('Wining rate of player 1.', fontsize=20)
    axs[2].set_ylabel('Tie rate.', fontsize=20)

    if style == 0:
        for i in range(3):
            axs[i].fill_between(x=np.arange(mean_n.shape[0]), y1=min_n[:, i], y2=max_n[:, i], alpha=0.2, color='r')
            if i==0:
                print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f___' % ('player_0', max(min_n[:,i]), max(mean_n[:,i]), max(max_n[:,i])))
                axs[0].set_ylabel('Wining rate of player 0.', fontsize=20)
            if i==1:
                print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f___' % ('player_1', max(min_n[:,i]), max(mean_n[:,i]), max(max_n[:,i])))
                axs[1].set_ylabel('Wining rate of player 1.', fontsize=20)
            if i==2:
                print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f' % ('tie', max(min_n[:,i]), max(mean_n[:,i]), max(max_n[:,i])))
                axs[2].set_ylabel('Tie rate.', fontsize=20)

            axs[i].set_xlabel('Iteration.', fontsize=20)
            axs[i].tick_params(axis="x", labelsize=20)
            axs[i].tick_params(axis="y", labelsize=20)
            axs[i].set_xticks([0, int(mean_n.shape[0] / 2), int(mean_n.shape[0])])
            axs[i].set_yticks([0, 0.5, 1])
        fig.savefig(out_dir + '/' + filename.split('.')[0] + print_info[0] + print_info[1] + print_info[2] + '.png')
    else:
        for event in group:
            axs[0].plot(event[:,0], linewidth=1)
            axs[1].plot(event[:,1], linewidth=1)
            axs[2].plot(event[:,2], linewidth=1)
            axs[0].set_ylabel('Wining rate of player 0.', fontsize=20)
            axs[1].set_ylabel('Wining rate of player 1.', fontsize=20)
            axs[2].set_ylabel('Tie rate.', fontsize=20)
        for i in range(3):
            axs[i].set_xlabel('Iteration.', fontsize=20)
            axs[i].tick_params(axis="x", labelsize=20)
            axs[i].tick_params(axis="y", labelsize=20)
            axs[i].set_xticks([0, int(len(event) / 2), len(event)])
            axs[i].set_yticks([0, 0.5, 1])
        fig.savefig(out_dir + '/' + filename.split('.')[0] + '.png')


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_robustness/MuJoCo/agent-zoo/self-play/SumoHumans-v0_latest_0.005')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/rl_robustness/MuJoCo/agent-zoo/self-play/SumoHumans-v0_latest_0.005')
    parser.add_argument("--filename", type=str, default='results.png')
    args = parser.parse_args()

    out_dir = args.out_dir
    log_dir = args.log_dir
    filename = args.filename

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, style=0)
    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, style=1)

