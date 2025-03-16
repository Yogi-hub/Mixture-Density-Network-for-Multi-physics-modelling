### Mixture Density Networks for Modeling Gas-Surface Interactions in Rarefied Flows

This repository implements a **Mixture Density Network (MDN)** using **Keras** and **TensorFlow** to model gas-surface interactions in rarefied gas flows. The model predicts **post-collision velocities** which is used to calculate **accommodation coefficients (ACs)** for gas molecules interacting with solid surfaces, specifically for **Argon-Gold (Ar-Au)** and **Hydrogen-Nickel (H₂-Ni)** systems under different thermal and flow conditions. The results are compared with **Molecular Dynamics (MD)** simulations and **Gaussian Mixture Models (GMM)** approach, to evaluate the effectiveness of MDNs in capturing gas-surface interactions.  

## Overview  

Rarefied gas flows exhibit **non-equilibrium effects**, making traditional continuum-based models inadequate for capturing gas-solid interactions at high **Knudsen numbers** (Kn > 0.1). **Molecular Dynamics (MD) simulations** provide detailed atomic-scale insights but are computationally expensive. This project explores the use of **MDNs** as a machine-learning-based alternative to traditional parametric scattering kernel models, such as the **Cercignani-Lampis-Lord (CLL) model** and **Gaussian Mixture Models (GMMs).**  

### Key Contributions  

- **Machine Learning for Scattering Kernels**: The MDN predicts probability distributions for post-collision velocities, capturing complex gas-surface interactions.  
- **Comparison with MD and GMM**: The predicted ACs are evaluated against MD simulation data and GMM-based models.  
- **Handling Rayleigh-Distributed Components**: Unlike GMMs, MDNs can directly model Rayleigh-distributed velocity components without additional transformations.  
- **Hyperparameter Optimization**: **Keras Tuner** is used to determine the optimal network architecture and number of Gaussian components.  

## Methodology  

### 1. Molecular Dynamics (MD) Simulations  

MD simulations generate training data by simulating gas-surface interactions at different temperatures and flow conditions.  

- **Gas Species**: Ar (monoatomic) and H₂ (diatomic).  
- **Solid Surfaces**: Gold (Au) for Ar-Au and Nickel (Ni) for H₂-Ni.  
- **Temperature Conditions**:  
  - **Isothermal**: Both walls at 300K.  
  - **Non-Isothermal**: Bottom wall at 300K, top wall at 500K.  
- **Flow Conditions**: No flow vs. flow scenarios.  

### 2. Calculation of Accommodation Coefficients (ACs)  

The ACs are computed by analyzing velocity correlations between impinging and reflected gas molecules. A **least squares fit** on the scatter plots of velocity components provides the AC values.  

\[
\alpha = 1 - \beta
\]

where \( \beta \) is the slope of the least squares fit line in the velocity correlation plots.  

### 3. Mixture Density Network (MDN) Implementation  

MDNs combine neural networks with mixture density models to predict the probability density function of post-collision velocities:  

\[
P(Y = y | X = x) = \sum_{k=1}^{K} \pi_k \cdot \phi(y, \mu_k(x), \sigma_k(x))
\]

where:  
- \( K \) is the number of Gaussian components.  
- \( \pi_k \) are mixture coefficients (sum to 1).  
- \( \mu_k(x) \) are the mean values.  
- \( \sigma_k(x) \) are the standard deviations.  

#### **MDN Model Structure (Keras)**  

- Input Layer  
- Fully Connected Dense Layers (with ReLU activation)  
- MDN Output Layer  
  - **Mean** (\(\mu_k\)) - Linear Activation  
  - **Standard Deviation** (\(\sigma_k\)) - ELU Activation + Small Constant  
  - **Mixture Coefficients** (\(\pi_k\)) - Softmax Activation  

### 4. Hyperparameter Optimization  

Hyperparameter tuning is performed using **Keras Tuner** with Bayesian Optimization. The best-performing configuration for the **H₂-Ni system (isothermal, no flow)** is:  

| Hyperparameter     | Optimal Value |
|--------------------|--------------|
| Number of mixtures (K) | 44 |
| Activation Function | ReLU |
| Optimizer | Adam |
| Number of Layers | 2 |
| Units per Layer | 32 |

## Results

### 1. Comparison of Molecular Dynamics (MD), GMM, and MDN

The MDN predictions closely match the ACs from MD simulations, improving upon **GMM-based models** in handling **Rayleigh-distributed** velocity components and offering better generalization.

#### **Isothermal (Ar-Au System, No Flow)**

| Setup | \( V_x \) | \( V_y \) | \( V_z \) | \( E_{trans} \) |
|-------|----------|----------|----------|-----------|
| **MD**    | 0.877  | 0.955  | 0.880  | 0.889     |
| **GMM**   | 0.879  | -      | 0.888  | 0.884     |
| **MDN**   | 0.866  | 0.950  | 0.887  | 0.901     |
| **% Change (GMM vs MD)** | 0.23  | -      | 0.91   | 1.13      |
| **% Change (MDN vs MD)** | 1.25  | 0.52   | 0.79   | 1.34      |

#### **Isothermal (H₂-Ni System, No Flow)**

| Setup | \( V_x \) | \( V_y \) | \( V_z \) | \( E_{trans} \) | \( E_{rot} \) |
|-------|----------|----------|----------|-----------|-----------|
| **MD**    | 0.960  | 0.875  | 0.965  | 0.584  | 0.650  |
| **GMM**   | 0.958  | 0.798  | 0.954  | 0.558  | 0.700  |
| **MDN**   | 0.946  | 0.885  | 0.965  | 0.603  | 0.681  |
| **% Change (GMM vs MD)** | 0.21  | 8.80   | 1.14   | 4.45   | 7.69   |
| **% Change (MDN vs MD)** | 1.45  | 1.14   | 0.00   | 3.25   | 4.76   |


### 2. Performance Advantages over GMM  

- **No Need for Liao Transformations**: MDNs naturally handle Rayleigh-distributed velocity components.  
- **Better Generalization**: MDNs predict ACs more accurately on unseen test data.  
- **Optimized Gaussian Components**: Unlike GMM, where the number of components is pre-set, MDNs determine the optimal \( K \) dynamically.  

## Dependencies  

To run this project, install the required dependencies:  

```bash
pip install numpy pandas seaborn matplotlib scikit-learn scipy tensorflow keras keras-tuner molmass tqdm
```

- **NumPy**: Numerical operations.  
- **Pandas**: Data handling.  
- **Seaborn & Matplotlib**: Visualization.  
- **Scikit-learn**: Preprocessing and evaluation.  
- **SciPy**: Scientific computing.  
- **TensorFlow & Keras**: Neural network training.  
- **Keras Tuner**: Hyperparameter tuning.  
- **Molmass**: Molecular mass calculations.  
- **TQDM**: Progress tracking.  

## Repository Structure  

- `src/`: Source code for data preprocessing, MDN implementation, and evaluation.  
- `data/`: Processed datasets from MD simulations.  
- `results/`: Plots and tables comparing MDN with MD and GMM.  
- `support.py`: Utility functions for data handling and model evaluation.  

## Conclusion and Future Work  

This study demonstrates the effectiveness of MDNs in modeling gas-surface interactions, providing a **data-driven alternative** to traditional scattering kernels. Future work will focus on:  

- **Direct prediction of ACs** instead of post-collision velocities.  
- **Handling flow conditions** in non-isothermal environments.  
- **Reducing overfitting** by training with more data.



## References  

For a complete list of references, please see the full report.  


