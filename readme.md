About
=====

Reversi reinforcement learning by [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) methods.

Environment
-----------

* Python 3.6.3
* tensorflow-gpu: 1.3.0
  * tensorflow==1.3.0 is also ok, but very slow. When `play_gui`, tensorflow(cpu) is enough speed.
* Keras: 2.0.8

Modules
-------

### Reinforcement Learning

This AlphaGo Zero implementation consists of three worker `self`, `opt` and `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.

### Evaluation

For evaluation, you can play reversi with the BestModel.  

* `play_gui` is Play Game vs BestModel using wxPython. 

Data
-----

* `data/model/model_best_*`: BestModel.
* `data/model/next_generation/*`: next-generation models.
* `data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.
  
If you want to train the model from the beginning, delete the above directories.

How to use
==========

Self-Play
--------

```bash
python src/reversi_zero/run.py self
```
### options
* `--new`: create new BestModel
* `--type mini`: use Mini Config, (see `src/reversi_zero/configs/mini.py`)

Trainer
-------

```bash
python src/reversi_zero/run.py opt
```

### options
* `--type mini`: use Mini Config, (see `src/reversi_zero/configs/mini.py`)

Evaluator
---------

```bash
python src/reversi_zero/run.py eval
```

### options
* `--type mini`: use Mini Config, (see `src/reversi_zero/configs/mini.py`)

Play Game
---------

```bash
python src/reversi_zero/run.py play_gui
```

### Note: Mac pyenv environment

`play_gui` uses `wxPython`.
It can not execute if your python environment is built without Framework.
Try following pip install option.

```bash
env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.6.3
```
