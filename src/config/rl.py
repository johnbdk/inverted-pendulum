# General Parameters
TRAIN_STEPS = 500_000
TEST_STEPS = 10_000

TRAIN_FREQ = 1/24
TEST_FREQ = 1/60

EPSILON_POLICY = {
    'epsilon_start' : 0.3,
    'epsilon_end' : 0.0,
    'epsilon_decay_steps' : 0.5 * TRAIN_STEPS
}

Q_LEARN = {
    'alpha' : 0.2,
}
