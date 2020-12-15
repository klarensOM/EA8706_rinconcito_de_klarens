"""
Defines functions that construct various components of a reinforcement learning
agent.
"""
from typing import List, Any

from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


def get_agent(env) -> DDPGAgent:
    """
    Generate a `DDPGAgent` instance that represents an agent learned using
    Deep Deterministic Policy Gradient. The agent has 2 neural networks: an actor
    network and a critic network.

    Args:
    * `env`: An OpenAI `gym.Env` instance.

    Returns:
    * a `DDPGAgent` instance.
    """
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))

    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=100000, window_length=1)

    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                    gamma=.99, target_model_update=1e-3)#random_process=random_process, 
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent



class SaveBest(Callback):
    """
    Store neural network weights during training if the current episode's
    performance is better than the previous best performance.

    Args:
    * `dest`: name of `h5f` file where to store weights.
    """

    def __init__(self, dest: str):
        super().__init__()
        self.dest = dest
        self.lastreward = -200
        self.rewardsTrace = []
   

    def on_episode_end(self, episode, logs={}):
        self.rewardsTrace.append(logs.get('episode_reward'))
        if logs.get('episode_reward') > self.lastreward:
            self.lastreward = logs.get('episode_reward')
            self.model.save_weights(self.dest, overwrite=True)



def train_agent(agent, env, steps=30000, dest: str='agent_weights.h5f'):
    """
    Use a `DDPGAgent` instance to train its policy. The agent stores the best
    policy it encounters during training for use later. Once trained, the agent
    can be used to exploit its experience.

    Args:
    * `agent`: A `DDPGAgent` returned by `get_agent()`.
    * `env`: An OpenAI `gym.Env` environment in which the agent will operate.
    * `steps`: Number of actions to train over. The larger the number, the more
    experience the agent uses to learn, and the longer training takes.
    * `dest`: name of `h5f` file where to store weights.
    """
    store_weights = SaveBest(dest=dest)
    agent.fit(env, nb_steps=steps, visualize=False, verbose=1, callbacks=[store_weights])



class ActionSequence(Callback):
    """
    Store the history of actions taken by the agent. Useful for evaluating the
    agent's performance. Stores actions in a list of lists of episodes:
        [
            [a1, a2, a3],        (episode 1)
            [a1, a2, a3, a4, a5] (episode 2)
        ]
    """

    def __init__(self, actions = []):
        self.actions = actions # all actions listed by episode
        super().__init__()

    def on_episode_begin(self, episode, logs={}):
        self.actionSeq = []    # actions in a single episode
    
    def on_episode_end(self, episode, logs={}):
        self.actions.append(self.actionSeq)
        
    def on_step_end(self, step, logs={}):
        self.actionSeq.append(logs.get('action'))



def test_agent(agent, env, weights='agent_weights.h5f', actions=[]) ->\
    List[List[Any]]:
    """
    Run the agent in an environment and store the actions it takes in a list.

    Args:
    * `agent`: A `DDPGAgent` returned by `get_agent()`.
    * `env`: An OpenAI `gym.Env` environment in which the agent will operate.
    * `actions`: A list in which to store actions.

    Returns:
    * The list containing history of actions.
    """
    store_actions = ActionSequence(actions)
    agent.load_weights(weights)
    agent.test(env, nb_episodes=1, visualize=False,verbose=0, callbacks=[store_actions])
    return actions