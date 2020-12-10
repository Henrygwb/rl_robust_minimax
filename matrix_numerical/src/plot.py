import os
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
            if keys is not None and value.tag not in keys:
                continue
            events.append(value.simple_value)
    return events


def load_tb_data(log_dir, keys=None):
    event_paths = find_tfevents(log_dir)
    pool = multiprocessing.Pool()
    events_by_path = pool.map(functools.partial(read_events_file, keys=keys), event_paths)
    return events_by_path


def dist(x, y, x0, y0):
    return (x - x0) ** 2 + (y - y0) ** 2


# self-play 
def read_and_draw(folder, out_dir, exp):

    fig, ax = plt.subplots(figsize=(10, 8))	
    nash_points = [(0.5, 0.5), (0, 0), (0, 0), (0, 0)]

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
            ns = ns + 0.6
        else:
            ns = ns + 0.5

    ax.plot(ns, linewidth=1, color='indigo')
    ax.set_xlabel('Training iteration.')
    if 'Match' in exp:
        ax.set_ylabel('Probability of player 1 playing head.')
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_ylabel('Value of x.')
        ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])

    ax.set_xticks([0, int(len(eve)/2), len(eve)])
    fig.savefig(
        out_dir + '/' + exp + '.png')

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
    folder = '/Users/Henryguo/Desktop/rl_robustness/matrix_numerical/agent-zoo'
    out_dir = folder
    for game in ['Match_Pennies_OPPO_Model_0', 'As_Match_Pennies_OPPO_Model_0', 'CC_OPPO_Model_0', 'NCNC_OPPO_Model_0']:
        read_and_draw(folder, out_dir, game)

