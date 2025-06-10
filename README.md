Meet TorchGo, 
A fully open-sourced AI trained with self-play reinforcement-learning to master the traditional Chinese board game Go.

About TorchGo:
The TorchGo models have two components:
* A deep convolution stack, similar to Google Deepmind's AlphaGo and AlphaZero
  * Learns a policy function across the board space as a distribution of likely best moves
  * A subsequent Value head learns to predict the likely outcome of the game [White wins: -1, Black wins: 1, Draw: 0]
* A Monte-Carlo Tree Search approach
  * Searches for the most visited candidate moves by simulating many games

This repo also includes the training and model infrastructure to modify and improve on TorchGo, 
as well as two versions of TorchGo - TorchGo-mini for 9x9 boards, and TorchGo-classic for traditional 19x19 boards.

It also includes a 'play vs human' mode, which shows: 
* the board
* when the game ends, returns the evaluation, territory, and move logs.
Optionally:
* the distribution of preferred moves
* the most frequently searched moves
* the TorchGo's 'evaluation' of the position on the fly

There's also an 'analysis' mode, which shows:
* the board
* TorchGo's evaluation
* TorchGo's policy and MCTS candidates
  * Returns top_k moves per position

How to build: 
TorchGo
git remote add origin https://github.com/ScottBiggs2/TorchGo.git
git branch -M main
git push -u origin main