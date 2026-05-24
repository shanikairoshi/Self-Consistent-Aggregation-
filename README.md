"# A2G_Journal_Version" 
# SCM-A2G-QFL: Self-Consistent Midpoint Aggregation for Quantum Federated Learning

This repository contains the implementation of **SCM-A2G-QFL**, a self-consistent midpoint aggregation framework for stable Quantum Federated Learning (QFL) with torus-valued Quantum Neural Network (QNN) parameters.

The proposed method addresses three coupled challenges in QFL:

1. **Client reliability heterogeneity** through QoS-aware trust weighting.
2. **Periodic QNN parameter geometry** through circular/torus-aware aggregation.
3. **Unstable server movement** through self-consistent midpoint refinement.

Unlike direct Euclidean aggregation, SCM-A2G-QFL does not simply average client parameters. Instead, it computes a QoS-weighted angular direction, evaluates the induced midpoint, recomputes client-supported directions from that midpoint, and accepts a global movement only when the movement becomes self-consistent.

---

## Main Contributions

- **QoS-aware aggregation:** client influence is adjusted using fidelity, latency, instability, and data-size information.
- **Torus-aware QNN aggregation:** QNN rotation parameters are treated as periodic angles rather than ordinary Euclidean vectors.
- **Self-consistent midpoint update:** the server update is refined until its own midpoint supports the accepted movement.
- **Controlled angular stress tests:** synthetic angular cases are used to demonstrate Euclidean seam failures and SCM stability.
- **Domain-general evaluation:** experiments are conducted on medical and financial binary classification datasets.
- **IBM hardware compatibility validation:** trained global QNN parameters can be evaluated on real IBM Quantum backends.

---

## Method Overview

At communication round \(t\), each selected client receives the current global QNN parameters \(\boldsymbol{\theta}_t\), trains locally, and returns local parameters \(\boldsymbol{\theta}_{i,t}\). The server then computes QoS-aware weights:

\[
q_{i,t}
=
\frac{F_{i,t}^{\alpha}}
{(\tau_{i,t}+\epsilon)^{\gamma}(V_{i,t}+\epsilon)^{\delta}},
\]

where \(F_{i,t}\) is fidelity or channel reliability, \(\tau_{i,t}\) is latency, and \(V_{i,t}\) is instability. The final weight is

\[
w_{i,t}
=
\frac{p_{i,t}q_{i,t}}
{\sum_{j\in\mathcal{S}_t}p_{j,t}q_{j,t}}.
\]

The wrapped local client direction is

\[
\mathbf{v}_{i,t}
=
\operatorname{wrap}_{[-\pi,\pi)}
(\boldsymbol{\theta}_{i,t}-\boldsymbol{\theta}_t).
\]

The initial movement is

\[
\mathbf{u}^{(0)}_t=\beta_t\sum_{i\in\mathcal{S}_t}w_{i,t}\mathbf{v}_{i,t}.
\]

SCM then refines the movement using the midpoint

\[
\mathbf{m}_t(\mathbf{u})=
\operatorname{wrap}_{[-\pi,\pi)}
\left(
\boldsymbol{\theta}_t+\frac{1}{2}\mathbf{u}
\right),
\]

and solves the fixed-point condition

\[
\mathbf{u}_t^\star
=
\beta_t\boldsymbol{\psi}_t(\mathbf{u}_t^\star).
\]

The next global model is

\[
\boldsymbol{\theta}_{t+1}
=
\operatorname{wrap}_{[-\pi,\pi)}
(\boldsymbol{\theta}_t+\mathbf{u}_t^\star).
\]

---

## Repository Structure

```text
SCM-A2G-QFL/
│
├── data/
│   ├── breast_lesions_usg/
│   └── baf/
│
├── src/
│   ├── aggregation/
│   │   ├── fedavg.py
│   │   ├── a2g.py
│   │   ├── mp_a2g.py
│   │   ├── scm_a2g.py
│   │   ├── fedcompass_lite.py
│   │   └── fedmrur_torus.py
│   │
│   ├── qnn/
│   │   ├── models.py
│   │   ├── circuits.py
│   │   └── optimizers.py
│   │
│   ├── data_preprocessing/
│   │   ├── preprocess_breast_lesions.py
│   │   └── preprocess_baf.py
│   │
│   ├── training/
│   │   ├── train_federated.py
│   │   ├── client_update.py
│   │   └── metrics.py
│   │
│   ├── hardware/
│   │   ├── ibm_level1_angle_validation.py
│   │   └── ibm_global_model_validation.py
│   │
│   └── utils/
│       ├── angle_utils.py
│       ├── plotting.py
│       └── csv_logger.py
│
├── experiments/
│   ├── run_breast_lesions.py
│   ├── run_baf.py
│   ├── run_controlled_angular_stress_tests.py
│   └── run_ibm_validation.py
│
├── results/
│   ├── figures/
│   ├── csv/
│   └── ibm/
│
├── requirements.txt
├── environment.yml
├── README.md
└── LICENSE
