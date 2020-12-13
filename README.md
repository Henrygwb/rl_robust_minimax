# rl_robustness

### Games:

Matrix and numerical games: Match Pennies, Asymmetric Match Pennies, f(x, y) = x^2 - y^2, f(x, y) = x^2y^2 - xy.

MuJoCo games: Kick And Defend, You Shall Not Pass, Sumo Humans, Sumo Ants.

### Code:

Training an RL agent with Minimax optimization and self-play.

Training an adversarial RL agent aganist the agent trained with Minimax optimization or self-play.

Retraining the victim agents against the adversarial agents.

### Code structure:
```common.py```: environments related functions.

```env.py```: define the game environments. 

```logger.py```: define the logger. 

```utils.py```: logger related functions. 

```ppo_selfplay.py```: define the selfplay related objects: training model, act model, learner, runner.

```ppo_minimax.py```: define the minimax play related objects: training model, act model, learner, runner.

```ppo_adv.py```: define the adversarial attack related objects: training model, act model, learner, runner.

```selfplay_train.py```: main function of training a selfplay agent.

```adv_train.py```: main function of training an adversarial agent.

```minimax_train.py```: main function of training a set of minimax agents.

```zoo_utils.py```: define the policy network models.
