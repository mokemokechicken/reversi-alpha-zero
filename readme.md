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
  * If `config.play.use_newest_next_generation_model = True`, this worker is useless. (It is AlphaZero method)

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

### install libraries with Anaconda
```bash
cp requirements.txt conda-requirements.txt
```
* Comment out lines for `jedi`, `Keras`, `parso`, `python-dotenv`, `tensorflow-tensorboard`, `wxPython` libraries
* Replace '-' with '_' for  `ipython-genutils`, `jupyter-*`, `prompt-toolkit` libraries
```bash
conda env create -f environment.yml
source activate reversi-a0
conda install --yes --file conda-requirements.txt
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

### Download Trained BestModel(If needed)

Download trained BestModel(trained by bellow Challenge 1) for example.

```bash
sh ./download_best_model.sh
```

### Download Trained the Newest Model(If needed)

Download trained the newest model(trained by Challenge 2) as BestModel.

```bash
sh ./download_newest_model_as_best_model.sh
```

Configuration
--------------

### 'AlphaGo Zero' method and 'AlphaZero' method

I think the main difference between 'AlphaGo Zero' and 'AlphaZero' is whether using `eval` or not.
It is able to change these methods by configuration.

#### AlphaGo Zero method

* `PlayConfig#use_newest_next_generation_model = False`
* `PlayWithHumanConfig#use_newest_next_generation_model = False`
* Execute `Evaluator` to select the best model.

#### AlphaZero method

* `PlayConfig#use_newest_next_generation_model = True`
* `PlayWithHumanConfig#use_newest_next_generation_model = True`
* Not use `Evaluator` (the newest model is selected as `self-play`'s model)

### policy distribution of self-play

In DeepMind's paper,
it seems that policy(π) data saved by self-play are distribution in proportion to pow(N, 1/tau).
After the middle of the game, the tau becomes 0, so the distribution is one-hot.

`PlayDataConfig#save_policy_of_tau_1 = True` means that the saved policy's tau is always 1. 

## other important hyper-parameters (I think)

If you find a good parameter set, please share in the github issues!

### PlayDataConfig

* `nb_game_in_file,max_file_num`: The max game number of training data is `nb_game_in_file * max_file_num`.

### PlayConfig, PlayWithHumanConfig

* `simulation_num_per_move` : MCTS number per move.
* `c_puct`: balance parameter of value network and policy network in MCTS.
* `resign_threshold`: resign threshold
* `parallel_search_num`: balance parameter(?) of speed and accuracy in MCTS.
  * `prediction_queue_size` should be same or greater than `parallel_search_num`.
* `dirichlet_alpha`: random parameter in self-play.

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
It evaluates BestModel and the latest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel. 

### options
* `--type mini`: use mini config for testing, (see `src/reversi_zero/configs/mini.py`)

Play Game
---------

```bash
python src/reversi_zero/run.py play_gui
```

### Note: Mac pyenv environment

`play_gui` uses `wxPython`.
It can not execute if your python environment is built without Framework.
Try following pyenv install option.

```bash
env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.6.3
```

For Anaconda users:
```bash
conda install python.app
pythonw src/reversi_zero/run.py play_gui
```

<img src="doc/img/play_gui.png">


When executed, ordinary reversi board will be displayed and you can play against BestModel.
After BestModel moves, numbers are displayed on the board.

* Top left numbers(1) mean 'Visit Count (=N(s,a))' of the last search.
* Bottom left numbers(2) mean 'Q Value (=Q(s,a)) on AI side' of the last state and move. The Q values are multiplied by 100.

View Training Log in TensorBoard
----------------

### 1. install tensorboard

```bash
pip install tensorboard
```

### 2. launch tensorboard and access by web browser 

```bash
tensorboard --logdir logs/tensorboard/
```

And access `http://<The Machine IP>:6006/`.

### Trouble Shooting

If you can not launch tensorboard by error,
try to create another new plain project which includes only `tensorflow` and `tensorboard`.

And

```bash
tensorboard --logdir <PATH TO REVERSI DIR>/logs/tensorboard/
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
* 1 game in Self-Play: about 100~200 sec (simulation_num_per_move = 500, thinking_loop = 2).
* 1 game in Evaluation: about 50 sec (simulation_num_per_move = 100, thinking_loop = 5).
* 1 step(mini-batch, batch size=512) in Training: about 2.3 sec.

Model Performance
===============

Challenge 1(AlphaGo Method)
------------

The following table is records of the best models.
For model performance evaluation,
sometimes I am competing with iOS app(https://itunes.apple.com/ca/app/id574915961) and the best model.
"Won the App LV x" means the model won the level at least once (regardless of the number of losses).

It takes about 2~3 hours to evaluate one model in my environment.
Therefore, if you divide the time taken by 3, you can see the approximate number of evaluation times.

I changed many parameters for try-and-error.

|best model generation|date|winning percentage to best model|Time Spent(hours)|note|
|-----|---|-----|-----|-----|
|1|-|-|-|　|
|2|2017/10/24|94.1%|-|　|
|3|2017/10/24|63.4%|13|　|
|4|2017/10/25|62.0%|3|　|
|5|2017/10/25|56.7%|8|　|
|6|2017/10/25|67.3%|7|　|
|7|2017/10/25|59.0%|3|　|
|8|2017/10/26|59.7%|6|　|
|9|2017/10/26|59.4%|3|　|
|10|2017/10/26|55.7%|5|　|
|11|2017/10/26|57.9%|9|　|
|12|2017/10/27|55.6%|5|　|
|13|2017/10/27|56.5%|7|　|
|14|2017/10/28|58.4%|20|　|
|15|2017/10/28|62.4%|3|　|
|16|2017/10/28|56.0%|11|　|
|17|2017/10/29|64.9%|17|　|
|18|2017/10/30|55.2%|19|　|
|19|2017/10/31|57.2%|33|　|
|20|2017/11/01|55.7%|12|　|
|21|2017/11/01|59.7%|20|　|
|22|2017/11/02|57.8%|19|　|
|23|2017/11/03|55.8%|15|　|
|24|2017/11/03|64.2%|12|　|
|25|2017/11/04|55.4%|21|　|
|26|2017/11/04|56.7%|6|　|
|27|2017/11/05|57.5%|11|　|
|28|2017/11/06|58.5%|15|　|
|29|2017/11/06|55.3%|5|　|
|30|2017/11/06|55.0%|8|　|
|31|2017/11/06|56.9%|5|　|
|32|2017/11/07|56.1%|9|　|
|33|2017/11/08|55.7%|22|　|
|34|2017/11/08|56.1%|3|　|
|35|2017/11/08|59.0%|3|　|
|36|2017/11/08|59.4%|3|　|
|37|2017/11/08|56.2%|9|　|
|38|2017/11/10|55.4%|52|Won the app LV9, LV10|
|39|2017/11/12|57.2%|29|　|
|40|2017/11/12|55.1%|12|Won the app LV11|
|41|2017/11/13|55.7%|14|Won the app LV12, 13, 14, 15, 16, 17. I can't win anymore.|
|42|2017/11/15|57.8%|18|Won the app LV18, 19|
|43|2017/11/15|55.8%|16|　|
|44|2017/11/16|57.5%|8|　|
|45|2017/11/16|56.2%|3|Won the app LV20|
|46|2017/11/18|55.6%|49|　|
|47|2017/11/19|55.9%|34|　|
|48|2017/11/19|59.4%|9|　|
|49|2017/11/20|55.9%|6|　|
|50|2017/11/22|56.0%|44|　|
|51|2017/11/26|55.8%|112|11/25 morning, changed c_puct from 3 to 1.5.|
|52|2017/11/26|59.7%|6|　|
|53|2017/11/28|56.2%|33|Won the app LV21|
|54|2017/11/29|59.0%|24|　|
|55|2017/12/01|56.6%|58|　|
|56|2017/12/03|58.1%|49|self-play: always save policy of tau=1|
|57|2017/12/04|55.1%|24|　|
|58|2017/12/05|55.9%|35|　|
|59|2017/12/06|55.4%|6|　|
|-|2017/12/06|-|-|implement https://github.com/mokemokechicken/reversi-alpha-zero/issues/13|
|60|2017/12/07|61.7%|25|　|
|61|2017/12/07|58.1%|3|Won the app LV21,22|
|62|2017/12/07|57.8%|11|　|
|-|2017/12/07|-|-|fix bug about virtual loss W|
|63|2017/12/08|57.5%|9|　|
|64|2017/12/08|56.0%|9|　|

Challenge 2 (AlphaZero Method)
------------

* use_newest_next_generation_model = True
* simulation_num_per_move = 400
* save_policy_of_tau_1 = True
* c_puct = 1
* save_model_steps = 200


|date|note|
|:---:|---|
|2017/12/15|Won the app LV1|
|2017/12/17|Won the app LV3,5,7,9|
|2017/12/18|Won the app LV11,13|
|2017/12/20|Won the app LV14|
|2017/12/21|Won the app LV15,16,17|
|2017/12/22|Won the app LV18,19,20,21,22,23,24,25|
|2017/12/23|Won the app LV26|
|2017/12/24|Won the app LV27,28|
|2017/12/25|no progress|
|2017/12/26|Lost the app LV29(0-2) (Model won 0, lost 2)|
|2017/12/27|Lost the app LV29(0-2) (Model won 0, lost 2)|
|2017/12/28|Model vs LV29: (2-4) (Model won 2, lost 4), Model vs GRhino LV2: (1-2)|
|2017/12/29|Model vs LV30: (1-2), Model vs GRhino LV2: (0-3)|
|2017/12/30|Model vs LV31: (0-2), Model vs GRhino LV2: (2-2)|
|2017/12/31|Model vs LV31: (0-2)|
|2018/01/04|Change max_file_num from 2000 to 300 (#26)|
|2018/01/05|Model vs LV31: (3-2), Model vs GRhino LV2: (4-1), vs Grhino LV3: (2-1)|
