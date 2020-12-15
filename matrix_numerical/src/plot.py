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
    return events_by_path


def read_events_file_minimax(events_filename, keys=None):
    events = []

    for event in tf.train.summary_iterator(events_filename):
        for value in event.summary.value:
            if keys is not None and value.tag != keys:
                continue
            events.append(value.simple_value)
    return events


def load_tb_data_minimax(log_dir, key, ne):
    event_paths = find_tfevents(log_dir)
    for event_tmp in tf.train.summary_iterator(event_paths[0]):
        continue
    values_all = []
    for value in event_tmp.summary.value:
        if key is not None and bool(key.match(value.tag)):
            values_all.append(value.tag)
    pool = multiprocessing.Pool()
    events_all = []
    for value in values_all:
        events_by_path = pool.map(functools.partial(read_events_file_minimax, keys=value), event_paths)
        events_all.append(events_by_path)
    events_all = [event for events in events_all for event in events]
    events_final = []
    print('Total numbers of runs %d.' %len(events_all))
    for event in events_all:
        event_mean = np.mean(event[1000:])
        if event_mean >= ne-0.1 and event_mean <= ne+0.1:
            events_final.append(event)
    print('Numbers of converged runs %d.' %len(events_all))

    return events_final


# self-play 
def plot_selfplay(folder, out_dir, exp):

    fig, ax = plt.subplots(figsize=(10, 8))	

    exp_folder = folder + '/' + exp
    if 'Match' in exp:
        key = 'head'
    else:
        key = 'v'

    event = load_tb_data(exp_folder, key)

    for i in range(len(event)):
        eve = event[i]
        ax.plot(eve, linewidth=0.5)

    ns = np.zeros((len(eve)))
    if 'Match' in exp:
        if 'As' in exp:
            if 'PLARYER_0' in exp:
                ns = ns + 0.6
            else:
                ns = ns + 0.4
        else:
            ns = ns + 0.5

    if 'As_CC' in exp:
        if 'PLARYER_0' in exp:
            ns = ns + 1

    ax.plot(ns, linewidth=1, color='indigo')
    ax.set_xlabel('Training iteration.', fontsize=20)
    if 'Match' in exp:
        ax.set_ylabel('Probability of the training player playing head.', fontsize=20)
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_ylabel('Value of x.', fontsize=20)
        ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])

    ax.set_xticks([0, int(len(eve)/2), len(eve)])
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.savefig(out_dir + '/' + exp[0:-13] + '.png')

    return 0


def plot_selfplay_all():

    folder = '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/agent-zoo'
    out_dir = folder
    for game in ['Match_Pennies_PLAYER_0_OPPO_Model_0', 'Match_Pennies_PLAYER_1_OPPO_Model_0',
                 'As_Match_Pennies_PLAYER_0_OPPO_Model_0', 'As_Match_Pennies_PLAYER_1_OPPO_Model_0'
                 'CC_PLAYER_0_OPPO_Model_0', 'NCNC_PLAYER_0_OPPO_Model_0']:
        plot_selfplay(folder, out_dir, game)

    import joblib
    folder = '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/agent-zoo'
    for game in ['As_CC_PLAYER_0_OPPO_Model_0', 'As_CC_PLAYER_1_OPPO_Model_0']:
        folders = os.listdir(os.path.join(folder, game))
        fig, ax = plt.subplots(figsize=(10, 8))

        for fo in folders:
            event = []
            for i in range(2241):
                i = str(i+1)
                i = i.zfill(5)
                if i == '02241':
                    loaded_params = joblib.load(folder+'/'+game+'/'+fo+'/checkpoints/model/'+i+'/model')
                    a = loaded_params['/pi/police:0']
                    w = loaded_params['/pi/w:0']
                    b = loaded_params['/pi/b:0']
                    mean = np.matmul(a, w) + b
                    event.append(mean[0, 0])
            # prepare data for attack.
            # for i in range(2241):
            #     i = str(i+1)
            #     i = i.zfill(5)
            #     if i == '02241':
            #         loaded_params = joblib.load(folder+'/'+game+'/'+fo+'/checkpoints/model/'+i+'/model')
            #         a = loaded_params['/pi/police:0']
            #         w = loaded_params['/pi/w:0']
            #         b = loaded_params['/pi/b:0']
            #         mean = np.matmul(a, w) + b
            #         loaded_params['/pi/police:0'] = mean
            #         loaded_params.pop('/pi/w:0')
            #         loaded_params.pop('/pi/b:0')
            #         if game == 'As_CC_PLAYER_0_OPPO_Model_0':
            #             joblib.dump(loaded_params, '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/victim-agent/As_CC/player_0/model_'+str(mean[0,0]))
            #         else:
            #             joblib.dump(loaded_params, '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/victim-agent/As_CC/player_1/model_'+str(mean[0,0]))
            if max(event) > 2 or min(event) < -2:
                continue
            ax.plot(event, linewidth=0.5)

        ns = np.zeros((len(event)))

        if 'PLARYER_0' in game:
            ns = ns + 1

        ax.plot(ns, linewidth=1.5, color='indigo')
        ax.set_xlabel('Training iteration.', fontsize=20)
        ax.set_ylabel('Value of x.', fontsize=20)
        ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])

        ax.set_xticks([0, int(len(event)/2), len(event)])
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        fig.savefig(folder + '/' + game[0:-13] + '.png')

    return 0


def plot_adv_attack(folder, out_dir, exp):

    fig, ax = plt.subplots(figsize=(10, 8))

    exp_folder = folder + '/' + exp
    if 'Match' in exp:
        key_adv = 'head'
        key_vic = 'victim head'
    else:
        key_adv = 'v'
        key_vic = 'victim v'

    event_adv = load_tb_data(exp_folder, key_adv)
    event_vic = load_tb_data(exp_folder, key_vic)

    for i in range(len(event_adv)):
        eve = event_adv[i]
        ax.plot(eve, linewidth=0.5)

    ax.plot(event_vic[0], linewidth=1, color='indigo')
    ax.set_xlabel('Training iteration.', fontsize=20)

    if 'Match' in exp:
        ax.set_ylabel('Probability of the adv player playing head.', fontsize=20)
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_ylabel('Value of x.', fontsize=20)
        ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])

    ax.set_xticks([0, int(len(eve)/2), len(eve)])
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.savefig(out_dir + '/' + exp + '.png')

    return 0


def plot_adv_attack_all():

    folder = '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/adv-agent-zoo'
    out_dir = folder
    games = os.listdir(folder)
    if '.DS_Store' in games:
        games.remove('.DS_Store')
    games_true = games.copy()
    for game in games:
        if 'png' in game:
            games_true.remove(game)
    for game in games_true:
        plot_adv_attack(folder, out_dir, game)

    return 0


def plot_minimax(folder, out_dir, exp):

    def save_fig(events, ne):
        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(len(events)):
            eve = events[i]
            ax.plot(eve, linewidth=0.5)

        ax.plot(ne, linewidth=1, color='indigo')

        ax.set_xlabel('Training iteration.', fontsize=20)

        if 'Match' in exp:
            ax.set_ylabel('Probability of the adv player playing head.', fontsize=20)
            ax.set_yticks([0, 0.5, 1])
        else:
            ax.set_ylabel('Value of x.', fontsize=20)
            ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])

        ax.set_xticks([0, int(len(eve)/2), len(eve)])
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        return fig

    exp_folder = folder + '/' + exp

    if 'Match' in exp:
        key_party_0 = re.compile('Head: \d th in 0')
        key_party_1 = re.compile('Head: \d th in 1')
    else:
        key_party_0 = re.compile('V: \d th in 0')
        key_party_1 = re.compile('V: \d th in 1')


    ne_0 = 0.0
    ne_1 = 0.0

    if 'Match' in exp:
        if 'As' in exp:
            ne_0 = ne_0 + 0.6
            ne_1 = ne_1 + 0.4
        else:
            ne_0 = ne_0 + 0.5
            ne_1 = ne_1 + 0.5

    if 'As_CC' in exp:
        ne_0 = ne_0 + 1

    event_0 = load_tb_data_minimax(exp_folder, key_party_0, ne_0)
    event_1 = load_tb_data_minimax(exp_folder, key_party_1, ne_1)

    ne_0 = np.zeros((len(event_0[0]))) + ne_0
    ne_1 = np.zeros((len(event_0[0]))) + ne_1

    fig = save_fig(event_0, ne_0)
    fig.savefig(out_dir + '/' + exp + '_party_0.png')
    plt.close()

    fig = save_fig(event_1, ne_1)
    fig.savefig(out_dir + '/' + exp + '_party_1.png')
    plt.close()

    return 0


def plot_minimax_all():

    folder = '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/agent-zoo/minimax'
    out_dir = folder
    games = os.listdir(folder)
    if '.DS_Store' in games:
        games.remove('.DS_Store')
    games_true = games.copy()
    for game in games:
        if 'png' in game:
            games_true.remove(game)
    for game in games_true[0:1]:
        plot_minimax(folder, out_dir, game)
    return 0



# def dist(x, y, x0, y0):
#     return (x - x0) ** 2 + (y - y0) ** 2

# i = 0
    # iter = 0
    # for x, y in zip(events_0, events_1):
    #     x = np.array(x)
    #     y = np.array(y)
    #     iter = len(x)
    #     dlist = dist(x, y, nash_point[0], nash_point[1])
    #     ax.plot(range(len(x)), dlist, color=colors[i])
    #     i += 1
    #
    # ax.set_xticks([0, int(iter * 1/2), iter])
    # ax.set_yticks([0, 0.1, 0.2])
    # ax.set_xlabel('Training_Iteration')
    # ax.set_ylabel('L2_distance')
    # plt.grid(True)
    # plt.show()
    # fig.savefig(out_dir + '/' + exp + '_dist.png')
    #
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # i = 0
    # d = 60
    # for x, y in zip(events_0, events_1):
    #     x = np.array(x)
    #     y = np.array(y)
    #     ax.plot(x, y, '-', color=colors[i])
    #     x = x[0::d]
    #     y = y[0::d]
    #     u = np.array([x[i+1] - x[i] for i in range(len(x)-1)])
    #     v = np.array([y[i+1] - y[i] for i in range(len(y)-1)])
    #     # scale = 5, width = 0.01
    #     plt.quiver(x[:-1], y[:-1], u, v, scale_units='xy', angles='xy', scale=5, width=0.01, color=colors[i])
    #     i += 1
    #
    # if exp == 'Match_pennies':
    #     label_x = 'P1(head)'
    #     label_y = 'P2(head)'
    # elif exp == 'TCP_off':
    #     label_x = 'P1(C)'
    #     label_y = 'P2(C)'
    # else:
    #     label_x = 'X'
    #     label_y = 'Y'
    #
    # if exp == 'Match_pennies' or exp == 'TCP_off':
    #     ax.set_xticks([0, 0.5, 1.0])
    #     ax.set_yticks([0, 0.5, 1.0])
    # else:
    #     ax.set_xticks([-0.5, 0, 0.5])
    #     ax.set_yticks([-0.5, 0, 0.5])
    #
    # ax.set_xlabel(label_x)
    # ax.set_ylabel(label_y)
    # plt.grid(True)
    # plt.show()
    #
    # fig.savefig(out_dir + '/' + exp + '_converge.png')


if __name__ == '__main__':

    plot_minimax_all()


