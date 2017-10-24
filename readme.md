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

Setup
-------
### install libraries
```bash
pip install -r requirements.txt
```

If you want use GPU,

```bash
pip install tensorflow-gpu
```

### set environment variables
Create `.env` file and write this.

```text:.env
KERAS_BACKEND=tensorflow
```

Basic Usages
------------

For training model, execute `Self-Play`, `Trainer` and `Evaluator`. 


Self-Play
--------

```bash
python src/reversi_zero/run.py self
```

When executed, Self-Play will start using BestModel.
If the BestModel does not exist, new random model will be created and become BestModel.

### options
* `--new`: create new BestModel
* `--type mini`: use mini config for testing, (see `src/reversi_zero/configs/mini.py`)

Trainer
-------

```bash
python src/reversi_zero/run.py opt
```

When executed, Training will start.
A base model will be loaded from latest saved next-generation model. If not existed, BestModel is used.
Trained model will be saved every 2000 steps(mini-batch) after epoch. 

### options
* `--type mini`: use mini config for testing, (see `src/reversi_zero/configs/mini.py`)
* `--total-step`: specify total step(mini-batch) numbers. The total step affects learning rate of training. 

Evaluator
---------

```bash
python src/reversi_zero/run.py eval
```

When executed, Evaluation will start.
It evaluates BestModel and the oldest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel. 

### options
* `--type mini`: use mini config for testing, (see `src/reversi_zero/configs/mini.py`)

Play Game
---------

```bash
python src/reversi_zero/run.py play_gui
```

<img src="doc/img/play_gui.png">


When executed, ordinary reversi board will be displayed and you can play against BestModel.
After BestModel moves, numbers are displayed on the board.

* Top left numbers(1) mean 'Visit Count (=N(s,a))' of the last search.
* Bottom left numbers(2) mean 'Q Value (=Q(s,a)) on AI side' of the last state and move. The Q values are multiplied by 100.

### Note: Mac pyenv environment

`play_gui` uses `wxPython`.
It can not execute if your python environment is built without Framework.
Try following pyenv install option.

```bash
env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.6.3
```

Tips and Memo
====

GPU Memory
----------

In my environment of GeForce GTX 1080, memory is about 8GB, so sometimes lack of memory happen.
Usually the lack of memory cause warnings, not error.
If error happens, try to change `per_process_gpu_memory_fraction` in `src/worker/{evaluate.py,optimize.py,self_play.py}`,

```python
tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
```

Less batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `NormalConfig`.

Training Speed
------

* CPU: 8 core i7-7700K CPU @ 4.20GHz
* GPU: GeForce GTX 1080
* 1 game in Self-Play: about 47 sec.
* 1 game in Evaluation: about 50 sec.
* 1 step(mini-batch, batch size=512) in Training: about 2.3 sec.