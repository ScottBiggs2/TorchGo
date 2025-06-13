Meet TorchGo, 
A fully open-sourced AI trained with self-play reinforcement-learning to master the traditional Chinese board game Go.

To-do: 
* implement the correct data augmentation (board flips) in training.py
* correctly implement batching MCTS calls to PolicyValueNet
* put PolicyValueNet on steroids: 
  * depth
  * training
  * LayerNorm?
  * Attention/transformer tricks to borrow? 
* add time tracking to training to estimate total resource draw required for full training

About TorchGo:
The TorchGo models have two components - the model and the search:

The model:
* A deep convolution stack, similar to Google Deepmind's AlphaGo and AlphaZero
  * 'Sees' boards at time, time-1 as well as boards with Gaussian estimated 'influence fields' at those times
  * Learns a policy function across the board space as a distribution of likely best moves
  * A subsequent Value head learns to predict the likely outcome of the game [White wins: -1, Black wins: 1, Draw: 0]

The search:
* A Monte-Carlo Tree Search approach
  * Draws n moves from the policy function recursively until temperature depth, 
    * Reduces by 1-cpunct pct per move past temperature depth 
  * Searches for the most visited candidate move in winning continuations by simulating many games

This repo also includes the training and model infrastructure to modify and improve on TorchGo, 
as well as two versions of TorchGo - TorchGo-mini for 9x9 boards, and TorchGo-classic for traditional 19x19 boards.

It also includes a 'play vs human' mode, which shows: 
* the board
* when the game ends, returns the evaluation, area scoring, and move logs.
Optionally:
* the distribution of preferred moves
* the most frequently searched moves
* the TorchGo's 'evaluation' of the position on the fly [White wins: -1, Tie: 0, Black wins: 1]

There's also an 'analysis' mode, which shows:
* the board
* TorchGo's evaluation
* Visualizes TorchGo's policy and MCTS candidates
  * Returns top_k moves per position

How to build: 
...