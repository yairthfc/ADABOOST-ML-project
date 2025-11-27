# ⚡ AdaBoost From Scratch — With Decision Stumps & Full Visualization

This repository implements the **AdaBoost ensemble algorithm from scratch**, using custom-built weak learners (Decision Stumps).  
It also includes:

- Full AdaBoost training pipeline  
- Decision surface plots  
- Weighted-sample visualization  
- Error curves for training and testing  
- Theoretical handwritten PDF  
- Synthetic dataset generator  

All code is written manually using NumPy — **no scikit‑learn models**.

---

# 📁 Project Structure

```
adaboost-from-scratch/
│
├── src/
│   ├── adaboost.py            # AdaBoost implementation
│   ├── decision_stump.py      # Weak learner used by AdaBoost
│   ├── adaboost_scenario.py   # Full experiment runner + plots
│   ├── loss_functions.py      # Misclassification error
│   ├── utils.py               # Plotly helpers (decision surface, etc.)
│   └── __init__.py
│
├── docs/
│   └── Answers.pdf            # Theoretical derivations & handwritten solutions
│
├── requirements.txt
└── README.md
```

---

# 🚀 Implemented Components

## 🔹 **AdaBoost Algorithm**
Source: `src/adaboost.py`  

Implements the full AdaBoost classifier using:

- Custom weak learner `DecisionStump`
- Weighted sample distribution \( D_t \)
- Weak learner error calculation
- Learner weight (alpha) calculation  
  \[
  lpha_t = rac{1}{2}\lnrac{1-\epsilon_t}{\epsilon_t}
  \]
- Updated sample weights  
- Final prediction by weighted sign of learners  
- Partial predictions for first \( T \) learners  
- Partial losses

The class also stores all historical weight vectors \( D_t \) for visualization.

---

## 🔹 **Decision Stump (Weak Learner)**
Source: `src/decision_stump.py`  

Implements a CART-style decision stump supporting:

- Single feature split  
- Threshold search along sorted feature values  
- Best sign (+1 / −1)  
- Misclassification-based threshold search  
- Predicting based on:  
  *“below threshold → −sign, above threshold → +sign”*

This stump is specifically suited for AdaBoost.

---

## 🔹 **Synthetic Dataset Generator & Experiment Runner**
Source: `src/adaboost_scenario.py`

Includes:

### ✔ Dataset generator  
Creates 2‑dimensional points in \([-1,1]^2\) and labels them according to a circle decision rule.  
Supports a configurable noise ratio.

### ✔ Training AdaBoost  
Trains AdaBoost with:

- Any number of learners (default 250)
- Weighted stumps
- Full train/test evaluation
- Partial losses for each \( t \)

### ✔ Visualizations  
- **Training vs Test error** over boosting iterations  
- **Decision surfaces** for:  
  - 5 learners  
  - 50 learners  
  - 100 learners  
  - 250 learners  
- **Best performing T** automatically identified  
- **Weighted-sample decision surface**  
  (sample size proportional to final weight \( D_T \))

---

# 📊 Visualization Examples

These correspond to the figures shown in the handwritten PDF (`docs/Answers.pdf`):

### 1️⃣ Training vs Test Error Curve  
Shows that:

- Train error decreases rapidly  
- Test error stabilizes  
- Overfitting depends on noise level  

### 2️⃣ Decision Surfaces for Different Numbers of Learners  
Shows boosting’s improvement over time.

### 3️⃣ Best Learner Count  
Automatic selection of the \( T \) minimizing test loss.

### 4️⃣ Weighted Samples Visualization  
Plots training points with marker sizes proportional to weights.

---

# 🧠 Theoretical Component

Included in: `docs/Answers.pdf`  
Contains handwritten derivations:

- AdaBoost weight update proof  
- Boosting intuition  
- Error analysis  
- VC-dimension context  
- Figures with loss curves and decision boundaries  
- Analysis for noise vs no-noise cases  

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

### Run all experiments:
```bash
python src/adaboost_scenario.py
```

This will:

- Generate datasets  
- Train AdaBoost  
- Produce evaluation plots  
- Display decision surfaces  
- Highlight best-performing \( T \)  
- Show weighted sample decision surface  

---

# ▶️ Manual Usage Example

```python
from adaboost import AdaBoost
from decision_stump import DecisionStump

model = AdaBoost(wl=DecisionStump, iterations=200)
model.fit(X_train, y_train)

pred = model.predict(X_test)
loss = model.loss(X_test, y_test)
```

---

# 🛠 Technologies Used

- Python  
- NumPy  
- Plotly  
- Custom ML implementations  

---

# 🎯 Learning Outcomes

This project demonstrates:

- Understanding of ensemble learning  
- Implementing AdaBoost from scratch  
- Weak learner vs strong learner behavior  
- Sample reweighting & exponential loss  
- Decision boundary evolution  
- Visualization of boosting dynamics  
- Bias/variance behavior under noise  

---

# 📘 License
MIT License.

---

# 🙌 Author  
**Yair Mahfud**
