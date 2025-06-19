# TorchGo

A fully open-source AI trained with self-play reinforcement learning to master the board game Go. This repository includes pre-trained models and comprehensive tools for playing, evaluating, and training Go AI systems.

## 🚀 Quick Start

Check out the interactive Colab demo: [TorchGo Demo](https://colab.research.google.com/drive/your-demo-link)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TorchGo.git
cd TorchGo

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if not included)
# Models are available in the models/ directory
```

## 🎯 Features

### Pre-trained Models
- **TorchGo-transformer-9x9**: Advanced GPT-style transformer model (recommended)
- **TorchGo-mini-test-4**: Deep convolutional neural network model

### Interactive Tools
- **Human vs Model**: Challenge TorchGo directly with optional policy/MCTS visualizations
- **Game Evaluation**: Analyze your games with AI-powered insights and move suggestions
- **Bot vs Bot**: Watch two AI models compete against each other
- **Match Series**: Compare model performance with statistical analysis

## 🏗️ Architecture

### Transformer Model (Primary)
Inspired by Google DeepMind's MuZero architecture:
- **Input**: 24-channel board representation (current, t-1, t-2 positions + Gaussian influence fields)
- **Architecture**: GPT-style transformer with shared trunk and specialized heads
- **Outputs**: 
  - Policy head: Move probability distribution
  - Value head: Game outcome prediction [-1, +1]
  - Pass head: Specialized pass move probability

### Convolutional Model (Legacy)
Based on AlphaGo/AlphaZero architecture:
- **Input**: 24-channel board representation with temporal and influence data
- **Architecture**: Deep residual convolutional network
- **Outputs**: Policy distribution and value prediction

### Search Algorithm
- **Batched Monte Carlo Tree Search (MCTS)**
- Efficient GPU utilization through batched operations
- Configurable playout counts and exploration parameters
- Temperature-based move selection for training vs. evaluation

## 📊 Model Performance

| Model | Board Size | Training Games | Training Epochs | Relative Strength |
|-------|------------|----------------|-----------------|-------------------|
| TorchGo-transformer-9x9 | 9×9 | 560+ | 168+ | Strongest |
| TorchGo-mini-test-4 | 9×9 | Limited | Limited | Baseline |

### MCTS Performance Guide
| Playouts | Quality | Time/Game | Use Case |
|----------|---------|-----------|----------|
| 64 | 85% | ~1 min | Fast evaluation |
| 128 | 92% | ~5 min | Standard play |
| 800 | 96% | ~30 min | Tournament level (used by AlphaGo) |

## 🛠️ Usage

### Training Your Own Model

1. **Configure hyperparameters** in `main.py`:
```python
num_iterations = 50
games_per_iteration = 10
num_playouts = 64
batch_size = 256
```

2. **Start training**:
```bash
python main.py
```

3. **Compare models** using the match series function:
```python
from play.bot_vs_bot import run_match_series
results = run_match_series(your_model, baseline_model, device, num_games=100)
```

### Playing Against TorchGo

```python
from play.human_vs_model import play_vs_net
from models.policy_value_transformer import PolicyValueTransformer

# Load model
BOARD_SIZE = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PolicyValueTransformer(BOARD_SIZE, d_model = 128,
                            nhead = 4, num_layers = 6,
                            num_head_layers = 2, dropout = 0.1)

model_path = "models/TorchGo-transformer-9x9.pth"
net.load_state_dict(torch.load(model_path))
net.to(device)

# Start game
play_vs_net(net, device, num_playouts=256, c_puct=1.5, displays=True)
```

### Evaluating Games

```python
from play.evaluate_game import review_game

# Analyze your game
review_game(net, device, top_k=5, board_size=9)
```

## 🔬 Technical Details

### Input Representation
- **6 base channels**: Current, t-1, t-2 board states (Black/White binary)
- **18 influence channels**: Gaussian influence fields at σ=1, 3, 6
- **Total**: 24 channels per position

### Training Process
- **Self-play generation**: MCTS-guided games with temperature scheduling
- **Experience replay**: Large buffer with prioritized sampling
- **Policy gradient**: Cross-entropy loss on move distributions
- **Value learning**: MSE loss on game outcomes

### Key Innovations
- **Batched MCTS**: Efficient GPU utilization
- **Influence fields**: Spatial awareness beyond immediate captures
- **Transformer architecture**: Long-range dependencies and attention
- **Modular design**: Easy model comparison and experimentation

## 🤝 Contributing

### Compute Contributions
1. Train a model using your preferred hyperparameters
2. Use `run_match_series()` to compare against TorchGo-transformer-9x9
3. If your model shows significant improvement, submit:
   - Demo notebook
   - Training configuration (`main.py`)
   - Model weights (`.pth` file)
   - Performance analysis

### Development Contributions
- Implement data augmentation (board rotations/flips)
- Extend to 19×19 boards
- Optimize MCTS algorithms
- Add new model architectures
- Improve evaluation metrics

## 📚 Resources

### Research Papers
- [MuZero Paper](https://arxiv.org/abs/1911.08265) - Core algorithm inspiration
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - MCTS fundamentals
- [AlphaGo Paper](https://www.nature.com/articles/nature16961) - Original breakthrough

### Related Projects
- [KataGo](https://github.com/lightvector/KataGo) - Advanced open-source Go AI
- [MuZero Explained](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a) - Implementation guide

## 📞 Contact

- **Email**: scottbiggs2001@gmail.com
- **LinkedIn**: [Scott Biggs](https://www.linkedin.com/in/scott-biggs-112970255)
- **GitHub Issues**: For bug reports and feature requests

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This project is actively maintained. For the latest updates and model releases, check the repository regularly.