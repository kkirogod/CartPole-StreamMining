# CartPole Control via Stream Mining: Learning from Experts in Real-Time

This repository contains a complete implementation of a **Stream Mining** solution to solve the classic **CartPole-v1** control problem. Instead of traditional reinforcement learning, this approach focuses on **imitation learning** by processing a continuous data stream of 100,000 instances provided by an expert.

## 📌 Project Overview
The goal is to develop a classifier capable of learning the expert's policy to maintain a pole in a vertical position indefinitely. The system assumes the task is successful if the pole remains upright for 500 consecutive time steps.

### Data Stream Characteristics
The stream consists of four numerical features and one binary target:
* **Cart Position**: Horizontal position of the cart.
* **Cart Velocity**: Speed of the cart.
* **Pole Angle**: Angle of the pole relative to the vertical.
* **Pole Angular Velocity**: Rotation speed of the pole.
* **Action (Target)**: Binary decision (0: Push Left, 1: Push Right).

## 🛠️ Methodology
The project follows the **Prequential Evaluation** (Test-then-Train) paradigm, where each sample is first used to test the model's prediction before being used for training. 

Three models from the **River** library were compared:
1. **Gaussian Naive-Bayes**: Used as the mandatory baseline.
2. **Hoeffding Tree**: A fast decision tree learner for data streams.
3. **Logistic Regression + StandardScaler**: A linear model wrapped in an incremental preprocessing pipeline to ensure feature scaling.

## 📊 Experimental Results

### Training Phase (Prequential Accuracy)
* **Logistic Regression:** 94.67%
* **Hoeffding Tree:** 93.70%
* **Naive Bayes (Baseline):** 80.46%

### Testing Phase (Gymnasium Environment)
Models were evaluated over 10 episodes in the `CartPole-v1` environment. 

| Model | Mean Reward | Std Dev |
| :--- | :---: | :---: |
| **Logistic Regression** | **500.00** | **± 0.00** |
| Hoeffding Tree | 276.10 | ± 224.17 |
| Naive Bayes (Baseline) | 52.70 | ± 8.04 |

## 🧠 Key Insights
* **Incremental Preprocessing:** The Logistic Regression model achieved a perfect score only after implementing a `StandardScaler`. This confirms that online normalization is critical for gradient-based optimizers in stream mining.
* **Stability vs. Accuracy:** While the Hoeffding Tree showed high accuracy during training, it exhibited significant instability in the physical simulation, indicating that minor classification errors in boundary states lead to system failure.
* **Stream Efficiency:** The project demonstrates that complex control tasks can be learned on-the-fly with minimal memory footprint using dedicated stream mining architectures.

## 🚀 Getting Started
1. **Install Dependencies**:
   ```bash
   pip install river gymnasium[classic-control] pygame numpy pandas
2. **Run the Project**:
   ```bash
   python codigo.py
