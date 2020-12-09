import numpy as np
from gym import Env
import multiprocessing as mp
from gym.spaces import Discrete, Tuple, Box
from baselines.common.vec_env.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


class MultiAgentEnv(Env):
    """
    Multi-agent env wrapper.
    """

    def __init__(self, num_agents):
        self.num_agents = num_agents
        assert len(self.action_space.spaces) == num_agents
        assert len(self.observation_space.spaces) == num_agents

    def step(self, action_n):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class MatrixGameEnv(MultiAgentEnv):
    """
    Two-player, normal-form games with symmetrically sized action space.
    """

    metadata = {'render.modes': ['human']}
    ACTION_TO_SYM = None

    def __init__(self, num_actions, payoff):
        """
        :params: payoff - shape num_actions num_actions payoff matrices
        """
        agent_space = Discrete(num_actions)
        overall_space = Tuple((agent_space, agent_space))
        self.action_space = overall_space
        self.observation_space = overall_space
        super().__init__(num_agents=2)

        payoff = np.array(payoff)
        assert (payoff.shape == (2, num_actions, num_actions))
        self.payoff = payoff

    def step(self, action_n):
        assert (len(action_n) == 2)
        i, j = action_n
        # observation of one player is the action of its opponent.
        self.obs_n = (j, i)
        rew_0 = self.payoff[0, i, j]
        rew_1 = self.payoff[1, i, j]
        # One step game.
        done = True
        return self.obs_n, (rew_0, rew_1), (done, done), dict()

    def reset(self):
        # Use a meaningless action as the start point.
        self.obs_n = (0, 0)
        return self.obs_n

    def close(self):
        # No-op, there is no close in this environment
        return

    def seed(self, seed=None):
        # No-op, there is no randomness in this environment.
        return


class FuncGameEnv(MultiAgentEnv):
    """
    Two-player, normal-form games with symmetrically sized action space.
    """

    metadata = {'render.modes': ['human']}
    ACTION_TO_SYM = None

    def __init__(self, num_actions, func, env_name, action_boundary):

        agent_space = Box(low=-action_boundary*np.ones(1), high=action_boundary*np.ones(1))
        overall_space = Tuple((agent_space, agent_space))
        observation_space = Discrete(num_actions)

        self.action_space = overall_space
        self.observation_space = Tuple((observation_space, observation_space))

        self.func = func
        self.env_name = env_name
        super().__init__(num_agents=2)

    def step(self, action_n):

        assert (len(action_n) == 2)
        i, j = action_n
        i = i[0]
        j = j[0]
        # observation is the other players move
        self.obs_n = (j, i)

        rew_0 = -self.func(i, j)
        rew_1 = self.func(i, j)
        done = True
        return self.obs_n, (rew_0, rew_1), (done, done), dict()

    def reset(self):
        # State is previous players action, so this doesn't make much sense;
        # just assume (0, 0) is start.
        self.obs_n = (0, 0)
        return self.obs_n

    def close(self):
        # No-op, there is no close in this environment
        return

    def seed(self, seed=None):
        # No-op, there is no randomness in this environment.
        return


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


class FakeSingleSpacesVec(VecEnv):
    """VecEnv equivalent of FakeSingleSpaces.
    :param venv(VecMultiEnv)
    :return a dummy VecEnv instance."""
    def __init__(self, venv, agent_id=0):
        observation_space = venv.observation_space.spaces[agent_id]
        action_space = venv.action_space.spaces[agent_id]
        super().__init__(venv.num_envs, observation_space, action_space)

    def reset(self):
        # Don't raise an error as some policy loading procedures require an initial observation.
        # Returning None guarantees things will break if the observation is ever actually used.
        return None

    def step_async(self, actions):
        raise NotImplementedError()

    def step_wait(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError()


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done[0]:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)
