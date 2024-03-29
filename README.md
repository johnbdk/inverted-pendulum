# inverted-pendulum
This project is about implemented a reinforcement learning approach in combination with artificial neural networks (ANN) and Gaussian processes (GP) to stabilize an inverted pendulum at a target position

## Gaussian Process

To train Sparse GP with architecture narx:
```bash
python main.py gp --train --sparse --nb ['num_of_past_inputs'] --na ['num_of_past_outputs'] --inducing ['num_inducing_points'] --sample [num_samples]
```

If --sample argument is left empty then GP uses the whole dataset, otherwise num_samples indicates how many samples from the dataset to be used for training/validating/testing.

To test Sparse GP with architecture narx:
```bash
python main.py gp --test --sparse --nb ['num_of_past_inputs'] --na ['num_of_past_outputs'] --fname ['filename'] --sim
```

The argument --fname already indicates how many inputs/outputs (i.e. nb/na) the GP used to be trained. Use this info to pass the --nb and --na arguments when typing the above command.

The argument --sim is a boolean flag indicating whether to do a simulation or prediction. If this argument is specified, the program does simulation, otherwise prediction.

NOTE: There is no need to add --inducing argument since the saved model already possesses this information.

To train/test GP (i.e. not sparse GP), just emit the --sparse argument from the above commands.

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

## Artificial Neural Network

Eg. to train ANN with architecture noe:
```bash
python .\main.py ann --train --model-arch noe
```

Eg. to test ANN with architecture noe best model:
```bash
python .\main.py ann --test --model-arch noe 
```
Eg. to run the prediction submission for ann narx:
```bash
python .\main.py ann --pred_submission --model-arch narx
```

Eg. to run the prediction submission for ann narx:
```bash
python .\main.py ann --sim_submission --model-arch narx
```

Eg. to run the grid search for ann narx:
```bash
python .\main.py ann --grid_search
```

Eg. to run the grid search evalution for ann narx:
```bash
python .\main.py ann --grid_eval
```
