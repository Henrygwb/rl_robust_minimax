import os
import joblib
from common import env_list
from scipy.special import softmax

env = 3
GAME_ENV = env_list[env]
VICTIM_INDEX = 0

for VICTIM_INDEX in [0, 1]:
    VICTIM_PATH_all = '../victim-agent/selfplay/'+GAME_ENV+'/player_'+str(VICTIM_INDEX)
    models = os.listdir(VICTIM_PATH_all)
    if '.DS_Store' in models:
        models.remove('.DS_Store')

    for model in models:
        params = joblib.load(os.path.join(VICTIM_PATH_all, model))
        mean = params['/pi/police:0']
        if env < 2:
            mean = softmax(mean)[0, VICTIM_INDEX]
        else:
            mean = mean[0, 0]
        VICTIM_PATH = os.path.join(VICTIM_PATH_all, model)
        SAVE_DIR = '../adv-agent-zoo/'+GAME_ENV + '_VictimIDX_' + str(VICTIM_INDEX) + '_VictimMODEL_' + model + '_VictimPARAM_' + str(mean)
        for i in range(4):
            os.system('python adv_train.py ' + ' --env '+str(env) + ' --victim_idx '+str(VICTIM_INDEX) + ' --victim_path '+VICTIM_PATH + ' --save_path ' + SAVE_DIR + ' > console_%s_%d.txt  &' %(model, i))

