# General Parameters
TRAIN_STEPS = 500_000
TEST_STEPS = 10_000

MAX_EPISODE_STEPS = 1000

SAVE_FREQ = 5          # episodes

AGENT_REFRESH = 1/60

TEST_CALLBACK_FREQ = 100

STATE_SPACE_MAP = {
    'q_learn' : 'discrete',
    'dqn' : 'continuous',
    'a2c' : 'continuous',
    'a2c_built' : 'continuous'
}

ACTION_SPACE_MAP = {
    'q_learn' : 'discrete',
    'dqn' : 'discrete',
    'a2c' : 'continuous',
    'a2c_built' : 'continuous'
}

EPSILON_PARAMS = {
    'epsilon_start' : 1.0,
    'epsilon_end' : 0.1,
    'epsilon_decay_steps' : 0.5*TRAIN_STEPS,
}

QLEARN_PARAMS = {
    'alpha' : 0.2,
    'gamma' : 0.99,
}

DQN_PARAMS = {
    'learning_rate' : 0.001,
    'gamma' : 0.99,
    'buffer_size' : 50_000,
    'batch_size' : 64,
    'hidden_layers' : [20, 20],
    'target_update_freq' : 10_000,
}

ACTOR_CRITIC_PARAMS = {
    'learning_rate' : 0.001,
    'gamma' : 0.99,
    'alpha_entropy': 0.5,
    'alpha_actor': 0.5,
    'rollout_length': 1000,
}
