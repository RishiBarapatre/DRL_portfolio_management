# Deep Reinforcement Learning for Automated Asset Allocation

**An Adaptive Portfolio Management System using CNNs and Proximal Policy Optimization (PPO)**

Traditional algorithmic trading systems often rely on static rules and handcrafted indicators, struggling to adapt to the speed and complexity of modern financial markets. This project introduces a deep reinforcement learning (DRL) framework designed not just to issue single-stock buy/sell signals, but to optimally distribute capital across a multi-asset portfolio.

By combining a Convolutional Neural Network (CNN) for temporal feature extraction with a Proximal Policy Optimization (PPO) actor-critic architecture, this system learns adaptive allocation strategies directly from simulated market interactions.

## 🚀 Key Features

* **Multi-Asset Allocation:** Optimizes portfolio weights across a universe of 50 equities rather than discrete binary trading actions.
* **Temporal Feature Extraction:** Utilizes a custom CNN to process 30-day rolling windows of technical indicators, capturing local momentum, volatility, and mean-reversion patterns.
* **Realistic Market Simulation:** Custom OpenAI Gym environment incorporating transaction costs (L1 penalty) and portfolio valuation based on logarithmic returns.
* **Stable Policy Optimization:** Leverages PPO to prevent overly aggressive policy updates, ensuring stable and consistent learning.

## 🧠 Architecture & Methodology

### 1. Feature Engineering
The system tracks 50 assets, calculating 5 core technical indicators for each (RSI, SMA, Bollinger Bands Upper/Mid/Lower). This generates a 250-dimensional feature space per timestep. To capture short-term dependencies, the data is structured into 3D tensors representing 30-day temporal windows.

### 2. CNN Feature Extractor
The 3D observation tensor is processed through convolutional layers that slide along the temporal dimension. This compresses the multi-asset market conditions into a dense, 128-dimensional latent state vector.

### 3. PPO Actor-Critic Network
* **The Actor:** Uses the 128-dimensional latent vector to parameterize a multivariate normal distribution, outputting a continuous vector of normalized portfolio weights.
* **The Critic:** Estimates the expected long-term return (value) from the same latent state to guide the Actor's updates.

## 📊 Backtesting Results

The DRL agent was evaluated against the NIFTY 50 benchmark index. The agent successfully learned to navigate shifting market conditions, achieving a portfolio performance that tracks and, during specific volatile regimes, outperforms the standard benchmark.

*For detailed quantitative results, Sharpe ratios, and drawdown analysis, please refer to the [Technical Report](./Docs/Technical_report.pdf) included in this repository.*

## 🛠 Tech Stack

* **Machine Learning:** PyTorch, Stable Baselines3 (PPO)
* **Environment:** OpenAI Gym (Custom `PortfolioEnv`)
* **Data Processing:** Pandas, NumPy, Scikit-learn (StandardScaler)
* **Visualization:** Matplotlib, TensorBoard

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RishiBarapatre/DRL_portfolio_management.git](https://github.com/RishiBarapatre/DRL_portfolio_management.git)
   cd DRL_portfolio_management