# inverted-pendulum
This project is about implemented a reinforcement learning approach in combination with artificial neural networks (ANN) and Gaussian processes (GP) to stabilize an inverted pendulum at a target position

## Artificial Neural Network

Eg. to train ANN with architecture noe:
```bash
python .\main.py ann --train --model-arch noe
```

Eg. to test ANN with architecture noe and a specific model:
```bash
python .\main.py ann --test --model-arch noe --model-file state_space.pth 
```

Eg. to train ANN with architecture narx and display detailed processing information on your screen with the flag verbose:
```bash
python .\main.py ann --train --model-arch narx --verbose
```

Eg. to RE-train ANN with architecture narx and display detailed processing information on your screen with the flag verbose:
```bash
python .\main.py ann --train --model-arch narx --model-path narx.pth --verbose
```

As you can see, on the last example we also specified the model name. This is because the model, first, needs to be loaded so as to save its current state and continue its training.

## Gaussian Process

Eg. to train (test) Full GP with architecture narx:
```bash
python .\main.py gp --train --nb [number_of_past_inputs] --na [number_of_past_outputs]
```

Eg. to train (test) Sparse GP with architecture narx:
```bash
python main.py gp --train --nb [number_of_past_inputs] --na [number_of_past_outputs] --sparse --inducing 10
```

## Reinforcement Learning

Train RL with q learning (by default / you can omit the --agent flag):
```bash
python src/main.py rl --train --agent q_learn
```

Train RL with DQN:
```bash
python src/main.py rl --train --agent dqn
```

Train RL with A2C (single target):
```bash
python src/main.py rl --train --agent a2c_built
```

To specify the multi_target task, add the following flag (this is only applied on the actor critic method):
```bash
    --multi_target
```

To use render, add the following flag:
```bash
    --render
```

To load a specific trained RL agent (for example, session A2C_built_Best_0) and run inference:
```bash
python src/main.py rl --test --agent a2c_built --load A2C_built_Best_0 --render
```