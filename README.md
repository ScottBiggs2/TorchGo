Meet TorchGo, 
An AI trained with self-play reinforcement-learning to master the traditional Chinese board game Go.

About:
The TorchGo model has two components:
* A deep convolution stack, similar to Google Deepmind's AlphaGo and AlphaZero
  * Learns a policy function across the board space as a distribution of likely best moves
  * A subsequent Value head learns to predict the likely outcome of the game
* A Monte-Carlo Tree Search approach
  * Searches for the most visited candidate moves by simulating many games

This repo also includes the training and model infrastructure to modify and improve on TorchGo. 
It also includes a 'play vs human' mode, which shows: 
* the board
* the distribution of preferred moves
* the most frequently searched moves
* the TorchGo's 'evaluation' of the position

How to build: 
TorchGo
git remote add origin https://github.com/ScottBiggs2/TorchGo.git
git branch -M main
git push -u origin main