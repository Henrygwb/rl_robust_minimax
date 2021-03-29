import os, sys, subprocess
import numpy as np
import gym
import roboschool
import tensorflow as tf
from baselines.common.policies import build_policy
from ppo_minimax import Model
from baselines.common.tf_util import get_session
from stable_baselines.common.vec_env import DummyVecEnv

def play(env, model):
    obs = env.reset()
    done = False
    while 1:
        a, _, _, _ = model.step(obs, S=None, M=done)
        obs, _, done,_ = env.step(a)

if len(sys.argv)==1:
    import roboschool.multiplayer
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pongdemo", want_test_window=True)
    for n in range(game.players_count):
        subprocess.Popen([sys.executable, sys.argv[0], "pongdemo", "%i"%n])
    gameserver.serve_forever()

else:
    player_n = int(sys.argv[2])

    env = gym.make("RoboschoolPong-v1")
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)
    env = DummyVecEnv([lambda: env])

    # set up the model
    sess = get_session()
    policy = build_policy(env, 'custom_mlp', value_network='copy')
    model = Model(policy=policy, nbatch_act=1, nbatch_train=1, ent_coef=0.0, vf_coef=0.0,
                      max_grad_norm=0.1, model_index=str(player_n)+'_'+str(0), sess=sess)
    sess.run(tf.global_variables_initializer())

    # load the model
    if player_n == 0:
        model_path = '/home/xkw5132/checkpoints/model_0_0/model'
    else:
        model_path = '/home/xkw5132/checkpoints/model_1_1/model'
    model.load(model_path)

    play(env, model)   # set video = player_n==0 to record video                                                                  
