# inverted-pendulum
This project is about implemented a reinforcement learning approach in combination with artificial neural networks (ANN) and Gaussian processes (GP) to stabilize an inverted pendulum at a target position

Eg. to train ANN with architecture noe:
```bash
python .\main.py ann --train --model-arch noe
```

Eg. to test GP with architecture narx:
```bash
python .\main.py gp --test --model-arch narx   
```

Eg. to test ANN with architecture ss and specific model:
```bash
python .\main.py gp --test --model-arch narx --model-file state_space.pth 
```

Eg. to train ANN with architecture narx and display detailed processing information on your screen with the flag verbose:
```bash
python .\main.py ann --train --model-arch narx --verbose
```

Eg. to RE-train ANN with architecture narx and display detailed processing information on your screen with the flag verbose:
```bash
python .\main.py ann --train --model-arch narx --model-path narx.pth --verbose
```

As you can see, on the last example we spacified the model name also. This is because we, first, need to load the model, save its current state and continue its training.

```bash
python main.py gp --sparse --inducing 10 --train
```