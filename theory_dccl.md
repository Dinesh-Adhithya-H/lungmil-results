# Dynamically Consistent Contrastive Learning (DCCL): Theory

*Draft — for internal reading and editing. Not for distribution.*

---

## 1. Motivation and Background

Self-supervised learning (SSL) for time series has made substantial progress by borrowing contrastive objectives from vision and NLP. Methods such as TS2Vec, TS-TCC, and TNC learn an encoder $E_\theta : \mathcal{X} \to \mathbb{R}^d$ such that representations at nearby timestamps are similar and representations at distant timestamps are dissimilar. This produces useful embeddings for downstream classification and regression tasks.

However, these methods impose **no constraint on how representations evolve over time**. Given an encoder $E_\theta$, the representations $r_t = E_\theta(x_t)$ and $r_{t+\Delta t} = E_\theta(x_{t+\Delta t})$ are related only by proximity in embedding space — there is no requirement that there exists a consistent operator mapping one to the other. Formally, the space of valid transitions

$$\mathcal{T} = \{ \phi : \mathbb{R}^d \times \mathbb{R}_{\geq 0} \to \mathbb{R}^d \mid \phi(r_t, \Delta t) \approx r_{t+\Delta t} \}$$

is left entirely unconstrained by the contrastive loss. Any function that maps $r_t$ to $r_{t+\Delta t}$ is equally valid from the perspective of $L_\text{NCE}$ alone.

This is in stark contrast to the physical world from which observations are drawn. If the underlying system evolves according to a dynamical law, then the sequence of representations $\{r_t\}$ must themselves trace a trajectory in $\mathbb{R}^d$ consistent with that law. Specifically, **the transition operators must compose**: evolving from $t$ to $t + \Delta t_1$ and then from $t + \Delta t_1$ to $t + \Delta t_1 + \Delta t_2$ must be equivalent to evolving directly from $t$ to $t + \Delta t_1 + \Delta t_2$.

We call this the **semigroup property** of the dynamics, and we argue that enforcing it as an explicit training objective — beyond local next-step prediction — yields representations that are geometrically more structured and empirically more useful for downstream tasks, particularly under irregular temporal sampling.

---

## 2. Mathematical Preliminaries

**Definition 2.1 (One-parameter family of operators).** Let $\{\phi_t\}_{t \geq 0}$ be a family of maps $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ indexed by $t \geq 0$.

**Definition 2.2 (Semigroup).** The family $\{\phi_t\}_{t \geq 0}$ is a **(one-parameter) semigroup** if:
1. $\phi_0 = \mathrm{Id}$ (identity at $t=0$)
2. $\phi_{s+t} = \phi_s \circ \phi_t$ for all $s, t \geq 0$ (composition law)

The composition law is the central constraint: integrating for time $s$ then for time $t$ is the same as integrating for time $s+t$ directly.

**Definition 2.3 (Group).** If additionally $\phi_{-t} = \phi_t^{-1}$ exists for all $t \geq 0$, i.e. the dynamics are invertible, then $\{\phi_t\}_{t \in \mathbb{R}}$ forms a **one-parameter group**.

**Theorem 2.4 (Picard-Lindelöf, informal).** Consider the ODE $\dot{r} = f_\theta(r, t)$. If $f_\theta$ is uniformly Lipschitz in $r$, then for any initial condition $r_0$ there exists a unique solution $r(t)$, and the flow map $\phi_t(r_0) = r(t)$ is a diffeomorphism — in particular, $\phi_t$ is invertible and $\{\phi_t\}_{t \in \mathbb{R}}$ is a group. Backwards integration is achieved by solving with $-f_\theta$.

*Remark.* The group property means that under a Lipschitz Neural ODE, one can in principle integrate backwards: given a high-risk representation at time $t_\text{late}$, integrate with $-f_\theta$ to recover the representation at an earlier biopsy that led to it. This enables counterfactual reasoning ("what did the patient look like when the trajectory diverged?") without retraining.

**Definition 2.5 (Dynamically Consistent Representation).** A pair $(E_\theta, \phi_\theta)$ is *dynamically consistent* with respect to an observed time series $\{(x_{t_i}, t_i)\}_{i=1}^T$ if:
1. $\phi_\theta$ is (approximately) a semigroup: $\phi_\theta(r, s+t) \approx \phi_\theta(\phi_\theta(r, s), t)$
2. $\phi_\theta(E_\theta(x_{t_i}), \Delta t) \approx E_\theta(x_{t_i + \Delta t})$ for all observed pairs

Condition 1 is the semigroup constraint — $\phi_\theta$ must compose correctly. Condition 2 is the dynamics prediction constraint — $\phi_\theta$ must track the actual observed trajectory. Both are necessary; neither implies the other.

---

## 2b. Connections to Physics and Dynamical Systems

The semigroup framework is not an abstraction invented for machine learning — it is the language in which classical physics describes time evolution, and borrowing it for representation learning gives DCCL a physical grounding that purely statistical SSL methods lack.

**Hamiltonian mechanics.** In conservative physical systems (no energy dissipation), time evolution is governed by Hamilton's equations $\dot{q} = \partial H / \partial p$, $\dot{p} = -\partial H / \partial q$. The resulting flow $\phi_t$ is a *symplectomorphism* — a volume-preserving diffeomorphism on phase space. Critically, $\{\phi_t\}_{t \in \mathbb{R}}$ forms a group: the system is time-reversible and energy-conserving. When a Neural ODE is used to model such a system, $L_\text{semi}$ enforces that the learned flow map approximates this symplectic structure. Precisely, for a Hamiltonian system the Jacobian $D\phi_t$ satisfies $D\phi_t^T J D\phi_t = J$ where $J$ is the symplectic matrix; $L_\text{semi}$ provides a weaker but computationally tractable surrogate for this constraint, encouraging the flow map to be globally consistent without explicitly parameterising it as symplectic.

**Dissipative systems and the semigroup (not group).** Biological systems are generically dissipative: entropy increases, information is lost, and disease progression is irreversible. A patient who develops chronic lung allograft dysfunction (CLAD) does not spontaneously recover. In such systems, the flow map $\phi_t$ is **not** invertible — there is no $\phi_{-t}$ because the system trajectory cannot be run backwards in a physically meaningful sense. This is precisely the distinction between a group (invertible, reversible dynamics) and a semigroup (forward-only, irreversible dynamics). The choice between group and semigroup for $\phi_\theta$ is therefore a modelling assumption about the underlying biology:

- **Semigroup ($t \geq 0$ only):** appropriate for irreversible disease states — CLAD, fibrosis, organ failure. No backwards integration is biologically meaningful.
- **Group ($t \in \mathbb{R}$):** appropriate when the representation captures reversible fluctuations — inflammation that resolves, drug effects that wash out. Backwards integration recovers the earlier state.

In practice, the Neural ODE with Lipschitz $f_\theta$ yields a group (Picard-Lindelöf), but the learned representation may still be approximately irreversible if the training data shows one-directional trajectories. $L_\text{semi}$ is agnostic to this choice — it enforces composition for $\Delta t > 0$ regardless.

**Lyapunov stability and attractors.** A fixed point $r^*$ of $\phi_\theta$ satisfies $\phi_\theta(r^*, \Delta t) = r^*$ for all $\Delta t$, equivalently $f_\theta(r^*, t) = 0$. Such fixed points are *attractors* of the dynamics — clinical states that are stable and self-sustaining. Lyapunov stability theory classifies fixed points by the eigenvalues of the Jacobian $Df_\theta(r^*)$: negative real parts imply local stability (the system returns to $r^*$ after small perturbations); positive real parts imply instability (the system diverges from $r^*$).

$L_\text{semi}$ helps the model learn attractors consistently. Without semigroup enforcement, a locally-consistent dynamics model might predict $r^*$ correctly for short horizons but drift away for long horizons — precisely because the composition law is not enforced. With $L_\text{semi}$, trajectories that approach $r^*$ must do so in a globally coherent manner, which naturally stabilises the attractor in the learned dynamics.

**Ergodic theory.** Under suitable conditions (ergodicity), long-time averages of $\phi_\theta$-trajectories converge to an *invariant measure* $\mu$ on $\mathbb{R}^d$: the distribution of $r_t$ as $t \to \infty$. The empirically observed bimodal structure in death-survival representation UMAPs — two well-separated patient clusters corresponding to survivors and non-survivors — is an instance of a system with two distinct invariant measures, one for each basin of attraction. $L_\text{semi}$ encourages the learned dynamics to reproduce this bistable structure consistently across timescales, not just locally.

---

## 2c. Group Theory Perspective

**Abstract algebra recap.** In abstract algebra, a *group* $(G, \cdot, e, \text{inv})$ consists of:
1. A set $G$ with a binary operation $\cdot : G \times G \to G$ (closure)
2. Associativity: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
3. An identity element $e \in G$: $e \cdot a = a \cdot e = a$
4. Inverses: for each $a \in G$, there exists $a^{-1} \in G$ with $a \cdot a^{-1} = e$

Dropping (3) and (4) gives a *semigroup*. Keeping (3) but dropping (4) gives a *monoid*. For the time-translation operators: the identity is $\phi_0 = \mathrm{Id}$ (zero time evolution does nothing), so the family $\{\phi_t\}_{t \geq 0}$ is at minimum a monoid. It is a semigroup in the standard usage of the dynamics literature (the identity axiom is often taken for granted). It is a group only when $\phi_{-t}$ is also defined, i.e. when dynamics are reversible.

**The time-translation group action.** The additive group $(\mathbb{R}, +)$ — or its sub-semigroup $(\mathbb{R}_{\geq 0}, +)$ — acts on the state space $\mathbb{R}^d$ via the map

$$\phi : \mathbb{R}_{\geq 0} \times \mathbb{R}^d \to \mathbb{R}^d, \quad (t, r) \mapsto \phi_t(r)$$

This is a *group action* (more precisely, a semigroup action): the action respects the group structure of $(\mathbb{R}_{\geq 0}, +)$ in the sense that $\phi_{s+t} = \phi_s \circ \phi_t$. The semigroup $\{\phi_t\}$ is the image of this action — a homomorphic image of $(\mathbb{R}_{\geq 0}, +)$ in the monoid of continuous endomorphisms of $\mathbb{R}^d$.

**Equivariance vs. invariance — a critical distinction.** Standard contrastive SSL learns *invariant* representations: the encoder maps inputs at nearby times to the same representation, $E_\theta(x_t) \approx E_\theta(x_{t+\delta})$ for small $\delta$. Invariance discards temporal information entirely — the representation is the same regardless of when the observation was made.

DCCL learns *equivariant* representations: the encoder and dynamics together satisfy

$$\phi_\theta(E_\theta(x_t), \Delta t) \approx E_\theta(x_{t + \Delta t})$$

This means the action of time-translation on inputs is *mirrored* by $\phi_\theta$ acting on representations. The representation changes in a structured, predictable way as time passes — it does not stay the same, but it changes according to a consistent law. Equivariance is strictly more informative than invariance:

- An invariant rep: $r_t = r_{t+100} =$ same vector. All temporal information is discarded. Useless for predicting events that unfold over time.
- An equivariant rep: $r_{t+100} = \phi_\theta(r_t, 100)$. The representation at $t+100$ is *determined* by the representation at $t$ and the dynamics. Temporal structure is preserved and predictable.

For downstream tasks that depend on temporal trajectory (mortality prediction, rejection risk, disease progression), equivariance is the correct inductive bias.

**Connection to Geometric Deep Learning (Bronstein et al., 2021).** The geometric deep learning programme unifies many deep learning architectures as instances of learning functions that are equivariant to symmetry groups acting on input data: CNNs are equivariant to spatial translations, GNNs are equivariant to node permutations, spherical CNNs are equivariant to rotations $SO(3)$.

DCCL extends this programme to **temporal symmetries**. The relevant group is the time-translation semigroup $(\mathbb{R}_{\geq 0}, +)$ acting on sequences. Existing geometric deep learning focuses overwhelmingly on *spatial* symmetries (acting on static data) — DCCL makes the case that *temporal* symmetries deserve the same treatment for sequential data. The key difference is that temporal symmetries are not discrete (unlike permutations) and are not compact (unlike rotations on a sphere), which requires the continuous-time formalism of ODEs rather than finite group representations.

---

## 2d. Contrastive Learning and Dynamics in Latent Space

To precisely position DCCL among existing methods, it is useful to lay out the spectrum from purely contrastive SSL (no dynamics) through predictive SSL (implicit dynamics) to explicit dynamical models (ODE/SSM), characterising what each enforces and misses.

**Standard contrastive SSL: learning invariance.** SimCLR, MoCo, and their time series adaptations (TS2Vec, TNC, TS-TCC) construct positive pairs through augmentation (for static data) or temporal proximity (for time series), then minimise $L_\text{NCE}$. The learned encoder $E_\theta$ is approximately *augmentation-invariant* (or time-locally-invariant). There is no dynamics model — no $\phi_\theta$ — and no constraint connecting $r_t$ to $r_{t+\Delta t}$ beyond InfoNCE proximity. Representations at different times are independently encoded, with no structured relationship.

**Predictive SSL: implicit dynamics.** Contrastive Predictive Coding (CPC, Oord et al. 2018) adds a summary model $g_\theta$ and per-horizon predictors $W_k$: $\hat{r}_{t+k} = W_k g_\theta(r_{\leq t})$. This is a form of implicit dynamics — the model learns to predict future representations — but it is NOT a dynamical system in the mathematical sense:

1. Separate networks $W_k$ for each horizon $k$ do not compose: $W_{k_1 + k_2} \neq W_{k_2} W_{k_1}$ in general. The model has no concept of integrating from $t$ to $t+1$ to $t+2$ using the same "circuit" — it uses entirely different networks for the 1-step and 2-step predictions.
2. JEPA (Joint-Embedding Predictive Architecture, LeCun 2022) uses a single predictor network but still with no semigroup constraint — the predictor maps $(r_t, \text{action/time})$ to $\hat{r}_{t+1}$ without requiring that applying it twice gives the same result as applying the 2-step predictor.

**Why prediction ≠ dynamics.** The distinction is fundamental. A network that correctly predicts $r_{t+1}$ from $r_t$ is not necessarily a dynamical system. It may use different computational pathways for $\Delta t = 1$ day vs $\Delta t = 7$ days — separate "circuits" that happen to predict correctly at their respective horizons but do not compose into a single consistent flow. A true dynamical system has ONE vector field $f_\theta(r, t)$ that governs ALL timescales simultaneously. $L_\text{semi}$ enforces this by requiring that the same $\phi_\theta$ applied in sequence gives the same result as $\phi_\theta$ applied directly over the combined interval.

**Linear dynamics in latent space.** The special case $\phi_\theta(r, \Delta t) = \exp(A \Delta t) r$ is the unique family of dynamics where the semigroup property holds analytically: $\exp(A(s+t)) = \exp(As)\exp(At)$ by the Baker-Campbell-Hausdorff theorem (or directly from the power series definition of the matrix exponential). This connects to several existing families:

- **Koopman operator theory** (Koopman 1931, Lusch et al. 2018): seeks a finite-dimensional linear representation of nonlinear dynamics. $A$ is the *Koopman generator* — the infinitesimal generator of the Koopman semigroup. Learning $A$ jointly with $E_\theta$ is equivalent to learning a Koopman approximation with a deep encoder.
- **Linear recurrent models (S4, Mamba, LRU):** these models parametrise state transitions as $h_t = \exp(A \Delta t) h_{t-1} + B u_t$, where $A$ is learned. They ARE implicitly learning a linear semigroup — DCCL makes this explicit and adds contrastive structure on top, replacing the sequence prediction loss with $L_\text{NCE} + L_\text{pred} + L_\text{semi}$.
- **State-space models (SSMs):** the $A$ matrix in discretised SSMs (e.g. $\bar{A} = (I - \Delta t A/2)^{-1}(I + \Delta t A/2)$ in bilinear discretisation) approximates $\exp(A \Delta t)$. DCCL-linear learns the same $A$ but with a richer training objective.

**Nonlinear dynamics.** Linear dynamics in $\mathbb{R}^d$ cannot represent bifurcations, limit cycles, or multiple attractors — all of which are biologically plausible (e.g. the bistable survivor/non-survivor UMAP structure). Neural ODE dynamics $dr/dt = f_\theta(r, t)$ are strictly more expressive and can in principle represent any smooth vector field. The cost is that the semigroup property no longer holds analytically — it must be enforced via $L_\text{semi}$.

**The spectrum of dynamics models:**

| Model | Dynamics | Semigroup | Contrastive | Irregular $\Delta t$ |
|-------|----------|-----------|-------------|---------------------|
| Linear SSM (S4, Mamba) | Linear $\exp(A\Delta t)$ | By construction | ❌ | ✅ |
| CPC | Linear per-horizon $W_k$ | ❌ Non-composing | ✅ | ❌ |
| Latent ODE (Rubanova 2019) | Neural ODE | Approximate | ❌ | ✅ |
| LaTiM (MICCAI 2024) | Neural ODE | Approximate | ✅ | ✅ |
| **DCCL-linear** | Linear $\exp(A\Delta t)$ | By construction | ✅ | ✅ |
| **DCCL-ODE** | Neural ODE | Explicit ($L_\text{semi}$) | ✅ | ✅ |

DCCL fills the bottom-right corner of this table: nonlinear dynamics, explicit semigroup enforcement, contrastive loss, and irregular time handling — all four simultaneously.

---

## 3. The Three Losses — Formal Definitions

Let $\mathcal{S} = \{(x_{t_i}, t_i)\}_{i=1}^T$ be an observed time series with potentially irregular gaps $\Delta t_i = t_{i+1} - t_i > 0$.

### 3.1 $L_\text{NCE}$: Temporal Contrastive Loss

Define a **temporal neighborhood** $\mathcal{N}(t, \delta) = \{s : |s - t| \leq \delta\}$ for a bandwidth $\delta > 0$.

- **Positive pairs**: $(r_t, r_s)$ with $s \in \mathcal{N}(t, \delta)$ from the same sequence
- **Negative pairs**: $(r_t, r_u)$ with $u \notin \mathcal{N}(t, \delta)$ or drawn from a different sequence

$$L_\text{NCE} = -\mathbb{E}_{t, s^+, \{u_k\}} \left[ \log \frac{\exp(\text{sim}(r_t, r_{s^+}) / \tau)}{\exp(\text{sim}(r_t, r_{s^+}) / \tau) + \sum_{k=1}^{K} \exp(\text{sim}(r_t, r_{u_k}) / \tau)} \right]$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau > 0$ is a temperature hyperparameter.

**What it enforces:** Local geometric structure in $\mathbb{R}^d$ — nearby timestamps cluster, distant timestamps separate. It does **not** constrain the transitions between $r_t$ and $r_{t+\Delta t}$.

### 3.2 $L_\text{pred}$: Dynamics Prediction Loss

$$L_\text{pred} = \mathbb{E}_{i} \left[ \| \phi_\theta(r_{t_i}, \Delta t_i) - r_{t_{i+1}} \|^2 \right]$$

where the expectation is over all adjacent pairs $(t_i, t_{i+1})$ in the observed sequence.

**What it enforces:** Local consistency of transitions — $\phi_\theta$ must map each observed representation to the next observed representation given the actual time gap $\Delta t_i$.

**Why it is insufficient alone:** $L_\text{pred}$ is a **local** constraint. It constrains $\phi_\theta$ only at observed adjacent pairs. A $\phi_\theta$ that achieves $L_\text{pred} \to 0$ on all adjacent pairs may still violate the composition law at non-adjacent pairs. Concretely:

> **Counterexample.** Let $\Delta t_1 = \Delta t_2 = 1$ day. Suppose $\phi_\theta(r, 1) = Wr + b$ (affine, step-size 1). This achieves $L_\text{pred} = 0$ for all adjacent pairs. But $\phi_\theta(r, 2) = Vr + c$ with $V \neq W^2$ and $c \neq Wb + b$ — a completely different affine map for the 2-day horizon. Then $L_\text{semi} > 0$: the model correctly predicts one day ahead but is globally incoherent across longer horizons. This is precisely the failure mode of CPC-style methods, which use separate linear projection heads per horizon $k$.

### 3.3 $L_\text{semi}$: Semigroup Self-Consistency Loss

For any triplet of observations $(t_{i-1}, t_i, t_{i+1})$ with gaps $\Delta t_1 = t_i - t_{i-1}$ and $\Delta t_2 = t_{i+1} - t_i$:

$$L_\text{semi} = \mathbb{E}_{i} \left[ \| \underbrace{\phi_\theta(r_{t_{i-1}},\, \Delta t_1 + \Delta t_2)}_{\text{direct path}} - \underbrace{\phi_\theta\!\left(\phi_\theta(r_{t_{i-1}},\, \Delta t_1),\, \Delta t_2\right)}_{\text{two-step path}} \|^2 \right]$$

**Generalisation to $n$-tuples.** The triplet is the minimal unit, but $L_\text{semi}$ extends naturally to arbitrary subsets. For any three indices $i < j < k$ drawn from the sequence:

$$L_\text{semi}^{(i,j,k)} = \| \phi_\theta(r_{t_i},\, t_k - t_i) - \phi_\theta\!\left(\phi_\theta(r_{t_i},\, t_j - t_i),\, t_k - t_j\right) \|^2$$

In practice, triplets are sampled uniformly from the sequence at each training step.

**What it enforces:** Global consistency — the composition law of the semigroup. Unlike $L_\text{pred}$, which operates only on adjacent pairs, $L_\text{semi}$ operates on any three observations regardless of whether they are adjacent. For irregular time series where observations may be missing (e.g. a patient missing a biopsy visit), $L_\text{semi}$ forces the dynamics to be coherent across the resulting larger gap.

### 3.4 Total Objective

$$\mathcal{L} = L_\text{NCE} + \lambda_1 L_\text{pred} + \lambda_2 L_\text{semi}$$

with $\lambda_1, \lambda_2 > 0$ as hyperparameters. The three losses are **complementary**:

| Loss | Constrains | Misses |
|------|-----------|--------|
| $L_\text{NCE}$ | Geometry of rep space | Nothing about transitions |
| $L_\text{pred}$ | Local transitions (adjacent) | Global composition across gaps |
| $L_\text{semi}$ | Global transition consistency | Geometry of rep space itself |

---

## 4. Two Dynamics Variants

### 4.1 Linear Dynamics (Koopman-Inspired)

$$\phi_\theta(r, \Delta t) = \exp(A \cdot \Delta t)\, r, \qquad A \in \mathbb{R}^{d \times d} \text{ learned}$$

where $\exp(\cdot)$ denotes the matrix exponential.

**Semigroup property — satisfied analytically:**

$$\exp(A(s+t)) = \exp(As) \cdot \exp(At) \quad \forall\, s, t$$

Therefore $L_\text{semi} = 0$ **by construction** for the linear variant, regardless of the learned $A$. This has two important consequences:

1. The linear variant serves as a **clean ablation baseline** — any benefit from $L_\text{semi}$ in the ODE variant is attributable purely to the need to enforce semigroup consistency in nonlinear dynamics.
2. The linear variant is **interpretable**: the eigenvalues of $A$ determine growth/decay rates; eigenvectors determine the directions of change. Stable modes (negative real part of eigenvalue) correspond to returning dynamics; unstable modes correspond to progressive deterioration.

**Connection to Koopman theory.** The Koopman operator framework seeks a linear representation of nonlinear dynamics: find a feature map $\psi$ and operator $\mathcal{K}$ such that $\mathcal{K}\psi(r_t) = \psi(r_{t+1})$. The linear dynamics variant is equivalent to learning a Koopman approximation jointly with the encoder, where $A$ is the Koopman generator satisfying $\mathcal{K}_{\Delta t} = \exp(A \Delta t)$.

**Group (invertibility).** Since $\exp(A \cdot (-\Delta t)) = \exp(A\Delta t)^{-1}$ (assuming $A$ is diagonalisable or using the general matrix exponential), the linear variant is always invertible — it forms a group, not just a semigroup. Backwards integration is exact and free.

### 4.2 Neural ODE Dynamics

$$\frac{dr}{d\tau} = f_\theta(r, \tau), \qquad \phi_\theta(r_0, \Delta t) = r_0 + \int_0^{\Delta t} f_\theta(r(\tau), \tau)\, d\tau$$

integrated numerically (e.g. Dormand-Prince RK45 via `torchdiffeq`).

**Semigroup property — NOT analytically satisfied by neural approximation.**

A continuous ODE with Lipschitz $f_\theta$ has an exact flow that IS a group (Picard-Lindelöf). However, numerical integration with step size $h$ introduces a per-step discretisation error of $O(h^p)$ (order $p$ method). Over a sequence of steps, these errors accumulate and violate the semigroup law:

$$\| \hat{\phi}_\theta(r, s+t) - \hat{\phi}_\theta(\hat{\phi}_\theta(r, s), t) \| = O(h^p \cdot \max(s,t)/h) = O(h^{p-1})$$

Therefore $L_\text{semi} > 0$ in expectation for any finite-step integrator, and its gradient does real corrective work during training — it pushes $f_\theta$ toward vector fields whose numerical integration is globally coherent, not just locally accurate.

**Group (Picard-Lindelöf).** If $f_\theta$ is Lipschitz, $\phi_\theta$ is invertible. Backwards integration:

$$\phi_\theta^{-1}(r_T, \Delta t) = r_T + \int_0^{\Delta t} (-f_\theta)(r(\tau), T - \tau)\, d\tau$$

This enables counterfactual queries: given a high-risk representation at time $T$, integrate backwards to recover the trajectory that produced it.

**Non-autonomous dynamics.** Using $f_\theta(r, \tau)$ (explicit time dependence) allows the vector field to change over absolute time — appropriate for non-stationary processes where the system dynamics themselves evolve (e.g. post-transplant immune reconstitution, drug washout periods). For stationary processes, $f_\theta(r)$ (autonomous) suffices and reduces the parameter count.

---

## 5. Why $L_\text{semi}$ Strictly Strengthens the Learning Signal

**Proposition 5.1 (Local consistency does not imply global consistency).** There exists a parametric family $\{\phi_\theta\}$ such that $L_\text{pred} = 0$ for all adjacent observed pairs yet $L_\text{semi} > 0$ for non-adjacent triplets.

*Proof sketch.* Construct $\phi_\theta$ as a lookup table: for each observed gap $\Delta t_i$, define a separate affine map $\phi_\theta(\cdot, \Delta t_i) = W_i \cdot + b_i$ trained to minimise $L_\text{pred}$. With sufficient capacity, $L_\text{pred} \to 0$. But for a non-observed gap $\Delta t_1 + \Delta t_2$, the model uses a separate map $W_j \neq W_2 W_1$, violating composition. $\square$

**Interpretation.** $L_\text{pred}$ is a **pointwise** constraint on the dynamics: it only supervises $\phi_\theta$ at the specific $(r_t, \Delta t)$ pairs that appear in the training data. $L_\text{semi}$ is a **functional** constraint: it requires that the dynamics operator is consistent as a function of time, not just at observed support points.

For **regular** time series (fixed $\Delta t$), adjacent-pair supervision is dense and the gap between local and global consistency is small — $L_\text{semi}$ adds little. For **irregular** time series (variable $\Delta t_i$, missing observations), large unsupervised gaps exist and $L_\text{semi}$ provides the only direct training signal for the dynamics at those timescales. This motivates the expected ablation pattern: $L_\text{semi}$ helps on PhysioNet (irregular) but not on HAR (regular fixed-rate).

---

## 6. Relationship to Existing Work

**vs. CPC (Oord et al., 2018).** CPC uses an autoregressive summary model $c_t$ and predicts future latents with separate affine maps $W_k$ per horizon $k$: $\hat{r}_{t+k} = W_k c_t$. These maps do NOT compose: $W_{k_1 + k_2} \neq W_{k_2} W_{k_1}$ in general — there is no dynamics operator, only independent per-horizon predictors. CPC is not a dynamical system. DCCL enforces composition explicitly.

**vs. TS2Vec (Yue et al., 2022).** Hierarchical contrastive at multiple timescales, but no dynamics model is learned. Representations $r_t$ and $r_{t+\Delta t}$ are related only through InfoNCE proximity — there is no $\phi_\theta$ and no composition constraint. DCCL adds a consistent dynamics layer on top of the contrastive objective.

**vs. Latent ODE (Rubanova et al., 2019).** Uses a VAE encoder with an ODE in latent space, trained with reconstruction loss. No contrastive loss, no semigroup enforcement — just local reconstruction. DCCL adds $L_\text{NCE}$ for representation geometry and $L_\text{semi}$ for global dynamics consistency.

**vs. LaTiM (Zeghlache et al., MICCAI 2024).** The closest existing work: combines SSL pretraining with Neural ODE for longitudinal medical imaging. However, LaTiM uses only next-step prediction loss — no semigroup consistency term. Additionally, LaTiM is evaluated only on retinal OCT (single domain, MICCAI scale), not on standard SSL4TS benchmarks. DCCL adds $L_\text{semi}$ and targets general time series.

**vs. FlowReg (ICLR 2026).** Aligns RL agent representations with Neural ODE trajectories as a regulariser. Key differences: (1) RL domain, not time series SSL; (2) no contrastive loss; (3) no explicit semigroup training objective — consistency is implicit via ODE uniqueness, not enforced as a composition loss with gradient.

---

## 7. Proposed Ablation Design

| Model | $L_\text{NCE}$ | $L_\text{pred}$ | $L_\text{semi}$ | $\phi_\theta$ |
|-------|---------------|----------------|----------------|--------------|
| E1 | ✅ | ❌ | ❌ | — |
| E2 | ✅ | ✅ | ❌ | ODE |
| E3 | ✅ | ✅ | ✅ | Linear $\exp(A\Delta t)$ |
| E4 | ✅ | ✅ | ✅ | Neural ODE |

**E1** is the TS2Vec-style baseline. **E2** is CPC-with-ODE (LaTiM-style). **E3** tests whether the linear dynamics + semigroup (which is analytically free) adds anything — it should not, since $L_\text{semi} = 0$ for linear $\phi_\theta$; if E3 > E2, the benefit comes from the linear dynamics model itself, not $L_\text{semi}$. **E4** is the full method.

**Expected pattern:**
- Irregular datasets (PhysioNet 2012, MIMIC-III): E4 > E2 >> E1, E3 ≈ E2
- Regular datasets (HAR, SleepEDF, ETTh1): E4 ≈ E2 > E1

If this pattern holds, it constitutes a **mechanistic confirmation** that $L_\text{semi}$ specifically addresses the irregular-gap failure mode of prediction-based methods.

---

## 8. Open Questions and Limitations

**Hyperparameters $\lambda_1, \lambda_2$.** No principled setting is known a priori. Recommended training schedule: (1) pretrain E1 ($L_\text{NCE}$ only) until convergence; (2) add $L_\text{pred}$ with $\lambda_1 = 1$ and fine-tune; (3) add $L_\text{semi}$ with $\lambda_2$ swept over $\{0.01, 0.1, 1.0\}$. This warm-up prevents $L_\text{semi}$ from dominating early training before $\phi_\theta$ has learned any useful dynamics.

**Triplet sampling.** For a sequence of length $T$, there are $\binom{T}{3}$ possible triplets. Uniform sampling over all triplets is feasible for short sequences (clinical biopsies, $T \leq 20$) but expensive for long sequences (ICU time series, $T \leq 48h$). Stratified sampling by gap size — ensuring a mix of short and long $(\Delta t_1, \Delta t_2)$ pairs — is recommended.

**Computational overhead.** Each triplet requires two ODE solves: one direct integration over $\Delta t_1 + \Delta t_2$, one two-step integration over $(\Delta t_1, \Delta t_2)$. This roughly doubles the ODE cost per training step compared to $L_\text{pred}$ alone. The linear dynamics variant has no overhead since both paths are matrix exponentials.

**Non-stationarity.** The semigroup law $\phi_{s+t} = \phi_s \circ \phi_t$ implicitly assumes **time-homogeneous** dynamics — the system evolves the same way regardless of absolute time. This may be violated for disease progression (transplant rejection risk changes over post-transplant years) or for drifting sensors. The non-autonomous ODE $f_\theta(r, \tau)$ partially addresses this by conditioning the vector field on absolute time, but the semigroup law then holds only for fixed anchor times, not in general. A time-varying semigroup (or $C_0$-semigroup with time-indexed generator) may be the correct formalisation for non-stationary settings — this is left for future work.

**Relationship to optimal transport.** The semigroup self-consistency loss can be interpreted as requiring that the learned dynamics map is consistent as an element of a function semigroup under composition. There may be a connection to displacement interpolation in Wasserstein space, where geodesics also satisfy a composition property — exploring this could yield theoretical bounds on representation quality.

---

*End of theory draft. All notation is provisional — harmonise with final implementation before submission.*
