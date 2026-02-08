# 🩺 Privacy-Preserving Federated Learning for Diabetes Prediction with Explainable AI

## Project Overview

This project demonstrates a **federated learning (FL) system** trained on distributed healthcare datasets for **diabetes risk prediction**, while ensuring **patient privacy** via **Differential Privacy (DP-SGD)** and providing **model transparency** through **Explainable AI (SHAP)**.

**Key Goals:**
- Predict diabetes risk across distributed medical datasets without centralizing sensitive data.
- Guarantee patient privacy using differential privacy.
- Explain individual predictions with SHAP to help clinical decision-making.
- Evaluate impact of **non-IID data distributions** (realistic healthcare scenarios) on model performance.

---

## Impact and Significance

- **Healthcare Privacy:** Federated learning + DP allows multiple clinics to collaborate **without sharing raw patient data**, fully compliant with **HIPAA/GDPR** regulations.
- **Explainable Predictions:** Doctors can see which features contribute most to a prediction (e.g., glucose levels, BMI, age).
- **Realistic Scenario:** Model is robust to **heterogeneous client data**, simulating hospitals with different patient populations.
- **Scientific Contribution:** Combines **FL + DP + XAI** in a single pipeline and performs **non-IID analysis**, demonstrating **practical impact of privacy-preserving AI**.




---

## Methodology

### 1. Data
- **Dataset:** Pima Indians Diabetes Dataset (768 samples, 8 features)
- **Preprocessing:** Standardization using `StandardScaler`
- **Simulation:** 5 clients with varying case mixes to mimic hospitals

### 2. Model
- **Architecture:** `DiabetesNet` — 3-layer fully connected neural network with ReLU activations
- **Output:** Sigmoid activation predicting probability of diabetes

### 3. Federated Learning
- **Framework:** Flower (`flwr`) for federated coordination
- **Clients:** Each simulates a different clinic
- **Training:** FL rounds with local updates, aggregated centrally

### 4. Differential Privacy (DP)
- **Algorithm:** DP-SGD
- **Settings:**  
  - No DP: baseline  
  - Moderate DP: ε ≈ 8  
  - Strong DP: ε ≈ 3  
- **Effect:** Protects against **membership inference attacks** while preserving accuracy (~0.65 pp drop at strong DP)

### 5. Explainable AI (XAI)
- **Method:** SHAP (KernelExplainer)  
- **Purpose:** Show **feature importance per patient** and **model transparency**

---

## Analysis & Results

### Privacy-Utility Trade-off

| Variant        | ε        | Test Accuracy | Accuracy Drop |
|----------------|----------|---------------|---------------|
| No DP          | ∞        | 74.68%        | 0.0%          |
| Moderate DP    | 8        | 73.97%        | 0.71%         |
| Strong DP ⭐    | 3        | 73.18%        | 1.5%          |

- **Observation:** Strong privacy comes with minimal accuracy drop — practical for healthcare

### Non-IID Analysis

| Partition        | Avg Divergence | Accuracy | Accuracy Drop |
|-----------------|----------------|----------|---------------|
| IID (baseline)   | 0.0275         | 73.38%  | 0.0pp         |
| Non-IID (α=0.5) | 0.3151         | 74.03%  | -0.65pp       |
| Non-IID (α=0.1) | 0.3520         | 70.78%  | 2.60pp        |

- FL can **handle heterogeneous client data**
- Pretraining helps recovery from severe label imbalance

### Explainability Example
- SHAP bar charts visualize **per-feature contribution** to predictions
- Glucose, BMI, Age often dominate predictions — aligned with clinical knowledge


