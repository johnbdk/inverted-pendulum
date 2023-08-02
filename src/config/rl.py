# General Parameters
TRAIN_STEPS = 500_000
TEST_STEPS = 10_000

MAX_EPISODE_STEPS = 1000

SAVE_FREQ = 5          # episodes

AGENT_REFRESH = 1/60

EPSILON_PARAMS = {
    'epsilon_start' : 0.3,
    'epsilon_end' : 0.0,
    'epsilon_decay_steps' : 0.5 * TRAIN_STEPS
}

QLEARN_PARAMS = {
    'alpha' : 0.2,
    'gamma' : 0.99
}

DQN_PARAMS = {
    'learning_rate' : 0.001,
    'gamma' : 0.99,
    'buffer_size' : TRAIN_STEPS // 10,
    'batch_size' : 64,
    'hidden_layers' : [24, 24],
    'target_update_freq' : TRAIN_STEPS // 8
}

ACTOR_CRITIC_PARAMS = {

}