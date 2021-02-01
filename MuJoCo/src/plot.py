import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_events_file_minimax(events_filename):
    def post_process(event, max_len):
        if event.shape[0] < max_len:
            len_diff = max_len - event.shape[0]
            event_tmp = np.random.normal(0, 0.01, (len_diff, ))
            event_tmp += np.mean(event[-200:])
            event = np.concatenate((event, event_tmp))
        elif event.shape[0] > max_len:
            event = event[0:max_len]
        return event[:, None]

    folders = os.listdir(events_filename)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    events_party_0 = []
    events_party_1 = []
    for folder in folders:
        log_file_party_0 = sorted([file for file in os.listdir(events_filename+'/'+folder) if 'Log_model' in file])
        log_file_party_1 = sorted([file for file in os.listdir(events_filename+'/'+folder) if 'Log_opp_model' in file])
        for file in log_file_party_0:
            events_party_0.append(np.loadtxt(os.path.join(events_filename, folder+'/'+file))[:, 1])
        for file in log_file_party_1:
            events_party_1.append(np.loadtxt(os.path.join(events_filename, folder+'/'+file))[:, 1])
    max_len = max([event.shape[0] for event in events_party_0])
    for i in range(len(events_party_0)):
        events_party_0[i] = post_process(events_party_0[i], max_len)
        events_party_1[i] = post_process(events_party_1[i], max_len)
    events_party_0 = np.array(events_party_0)
    events_party_1 = np.array(events_party_1)
    if 'YouShallNotPass' in events_filename:
        events_party_tie = np.zeros_like(events_party_0)
    else:
        events_party_tie = 1 - events_party_0 - events_party_1
    return np.concatenate((events_party_0, events_party_1, events_party_tie), axis=-1)


def read_events_file(events_filename):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            event = np.loadtxt(os.path.join(events_filename, folder+'/'+'Log.txt'))[:, 1:]
            events.append(event)
    max_len = max([event.shape[0] for event in events])
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
def plot(log_dir, out_dir, filename, style, selfplay):
    print_info = []
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(45, 15))
    if selfplay:
        group = read_events_file(log_dir)
    else:
        group = read_events_file_minimax(log_dir)
    mean_n = np.mean(group, axis=0)
    min_n = np.min(group, axis=0)
    max_n = np.max(group, axis=0)
    axs[0].set_ylabel('Wining rate of player 0.', fontsize=20)
    axs[1].set_ylabel('Wining rate of player 1.', fontsize=20)
    axs[2].set_ylabel('Tie rate.', fontsize=20)

    if style == 0:
        for i in range(3):
            axs[i].fill_between(x=np.arange(mean_n.shape[0]), y1=min_n[:, i], y2=max_n[:, i], alpha=0.2, color='r')
            axs[i].plot(mean_n[:, i], linewidth=1, color='r')
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
        fig.savefig(out_dir + filename + print_info[0] + print_info[1] + print_info[2] + '.png')
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
        fig.savefig(out_dir + filename + '.png')


def plot_all(folder, selfplay):

    out_dir = folder
    games = os.listdir(folder)
    if '.DS_Store' in games:
        games.remove('.DS_Store')
    games_true = games.copy()
    for game in games:
        if 'png' in game:
            games_true.remove(game)
    for game in games_true:
        plot(folder+game, out_dir, game, 0, selfplay)
        plot(folder+game, out_dir, game, 1, selfplay)
    return 0


# main function
if __name__ == "__main__":
    folder = '/Users/Henryguo/Desktop/rl_robustness/MuJoCo/agent-zoo/minimax/'
    # plot_all(folder, False)

    out_dir = folder
    games = os.listdir(folder)
    if '.DS_Store' in games:
        games.remove('.DS_Store')
    games_true = games.copy()
    for game in games:
        if 'png' in game:
            games_true.remove(game)
    print(games_true)
    for game in games_true:
        result = os.listdir(folder+game)
        if '.DS_Store' in result:
            result.remove('.DS_Store')
        result_true = result.copy()
        for rl in result:
            if 'png' in rl or 'mp4' in rl:
                result_true.remove(rl)
        for rl in result_true:
            for models_1 in ['model_0', 'model_1', 'opp_model_0', 'opp_model_1']:
                models = os.listdir(folder+game+'/'+rl+'/checkpoints/'+models_1)
                if '.DS_Store' in models:
                    models.remove('.DS_Store')
                if len(models) < 100:
                    continue
                else:
                    for model in models:
                        if int(model) < 720:
                            os.system('rm -r '+folder+game+'/'+rl+'/checkpoints/'+models_1+'/'+model)

