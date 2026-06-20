# A Toy Model of Superposition — a statistical-physics view

**How can a neural network remember more things (features) than it has room (neurons, dimensions) for? By superposition - the same way packed spheres do.**

This repository builds the smallest neural network that exhibits *superposition*, reproduces the key results from scratch, and then analyses them with the tools of statistical physics: order parameters, phase transitions, and energy minimisation. Everything runs on a laptop CPU in a few minutes.

### Based on Anthropic's "Toy Models of Superposition"

Everything here is built on **Elhage et al. (2022), *Toy Models of Superposition*** (Transformer Circuits Thread, <https://transformer-circuits.pub/2022/toy_model/index.html>). The model, the feature-dimensionality metric, the quantized polytope geometry, and the phase-change picture all originate in that work. This repository is an independent reimplementation of its core, re-examined through a statistical-mechanics lens.

### Scope

A compact, self-contained reproduction of the paper's *core* results plus an explicit statistical-physics analysis — deliberately narrow, not a full replication.

- **Reproduced from scratch** — the ReLU output model; the emergence of superposition with sparsity ($n=5$, $m=2 \to$ pentagon); the feature-dimensionality metric and its quantized values; a sparsity-driven phase diagram.
- **Added, extending the paper's own framing** — an explicit packing/frustration *energy* whose minimum the trained network attains exactly (a Thomson-problem reading of the geometry), and an order-parameter presentation of the transition with a derived scaling form (Section 5, Appendix).
- **Out of scope** — results that are in the paper but not attempted here: the linear-model contrast, computation-in-superposition, correlated/anti-correlated features, higher-dimensional polytopes, learning dynamics, and the connection to real trained networks.


---

## 1. Abstract

A neural network stores information inside a fixed number of internal "slots" (think of them as the axes of a coordinate system). Common sense says it can store at most one piece of information per slot. **Superposition** is the surprising trick where a network stores *many more* pieces of information than it has slots, by laying them down as *overlapping* directions that share the slots. This works only because the information is **sparse** — most pieces are "silent" at any given moment, so the overlaps rarely cause confusion. Below we watch a network discover this trick on its own, and we show that the shapes it settles into are exactly the shapes a physicist would predict from an energy-minimization argument.

### A few terms, defined once

- **Feature**: one independent piece of information the network might want to represent (e.g. "the image contains a wheel"). We use *n* features.
- **Dimension / slot**: one axis of the network's internal storage space. We use *m* of them, with **m < n** — fewer slots than features. That scarcity is the whole point.
- **Sparsity**: the fraction of the time a feature is *off* (exactly zero). Sparsity 0 = every feature is always on ("dense" world); sparsity near 1 = features almost never appear together ("sparse" world). This is our master control knob.
- **Superposition**: storing more features than dimensions by using overlapping, non-perpendicular directions.

---

## 2. Motivation — why superposition is worth understanding

Modern language models almost certainly rely on superposition heavily: they represent far more concepts than they have neurons. That makes it a central obstacle for **interpretability** (the effort to understand what a network has actually learned), because a single neuron ends up entangled in many unrelated concepts. A toy system small enough to solve completely is the natural place to build intuition before tackling it in real networks.

---

## 3. Why this is really a physics problem

If you must place *n* feature directions in only *m* dimensions, they cannot all be mutually perpendicular — there isn't enough room. So they interfere. Each pair of features that points in similar directions pays a penalty (they get confused for one another). The network wants to arrange the directions to make the *total* penalty as small as possible.

That is a **frustration problem**, and physicists have studied its cousins for over a century:

- It is the **Thomson problem**: place electrons on a sphere so that their mutual repulsion is minimized. The solutions are beautiful regular shapes.
- It is **sphere packing** and **spin frustration**: many parts, each wanting to avoid the others, settling into an ordered compromise.

And because we have a control knob (sparsity), we can ask the question at the heart of statistical mechanics: **does the system change phase as we turn the knob?** It does — sharply — and the ordered phases are regular polygons.

This is home territory for a statistical physicist — frustration, packing, and phase transitions are everyday tools there — and the rest of this repository applies them to superposition directly.

The original paper already thinks this way — it has a section on *phase changes* and discusses the geometry as uniform polytopes / packing. What this repo adds is to make that framing fully explicit and quantitative: a single order parameter swept across a control parameter, and a direct measurement that the learned geometry minimizes a packing energy (Section 5).

---

## 4. The model

The network is a deliberately minimal **autoencoder** (a network trained to copy its input to its output through a narrow bottleneck). It compresses $n$ features into $m < n$ dimensions and tries to reconstruct them:

$$h = W x \in \mathbb{R}^{m}, \qquad \hat{x} = \mathrm{ReLU}\!\left(W^{\top} h + b\right) = \mathrm{ReLU}\!\left(W^{\top} W x + b\right).$$

Here $W \in \mathbb{R}^{m \times n}$ is a single weight matrix, $b \in \mathbb{R}^{n}$ a bias, and $\mathrm{ReLU}(z) = \max(0, z)$ applied elementwise. **Column $i$ of $W$, written $W_i \in \mathbb{R}^{m}$, is the direction the network uses to store feature $i$** — its *representation vector*. The ReLU is what lets the network clip away the small interference from overlapping features.

**The data (the sparse world).** Each example $x \in \mathbb{R}^{n}$ has independent features; feature $i$ is off with probability $S$ (the sparsity) and otherwise uniform on $[0,1)$:

$$x_i = \begin{cases} 0 & \text{with probability } S, \\ u_i,\ \ u_i \sim \mathcal{U}[0,1) & \text{with probability } 1-S. \end{cases}$$

We call $p \equiv 1 - S$ the **density** (the probability a feature is on). The model never sees a fixed dataset — it learns the *statistics* of this world.

**The loss.** An importance-weighted mean squared reconstruction error, with optional per-feature importances $I_i$ (we use a geometric decay $I_i = r^{\,i-1}$, or $r=1$ for uniform):

$$L = \mathbb{E}_{x}\left[\, \sum_{i=1}^{n} I_i \,\bigl(x_i - \hat{x}_i\bigr)^2 \right].$$

The full engine is ~200 lines in [`src/superposition/`](src/superposition).

---

## 5. Results

### Experiment 1 — superposition emerges as the world gets sparser

`python experiments/01_superposition_emergence.py`

We use the tiniest interesting model: **n = 5 features, m = 2 dimensions**, so we can *draw* the storage space on a flat page. Each arrow is one feature's representation vector; the bar below counts how many dimensions each feature effectively occupies.

![Emergence of superposition](figures/01_superposition_emergence.png)

- **Dense world (sparsity 0):** the network can only afford **2** of the 5 features and stores them on **perpendicular axes** — no overlap, no superposition. The other three features are dropped.
- **As sparsity rises:** the network squeezes in more features. At intermediate sparsity it stores **4** features as two perpendicular pairs (a "plus" sign).
- **Sparse world (sparsity ≈ 0.97):** it stores **all 5** features as a perfect **regular pentagon**. The directions overlap, but since features rarely co-occur, the overlap rarely costs anything.

This is the canonical "Toy Models of Superposition" result, reproduced from scratch.

### Experiment 2 — a phase diagram

`python experiments/02_phase_diagram.py`

Now a bigger model (**n = 40 features, m = 10 dimensions**). We pick an **order parameter** — a single number summarising the macroscopic state — and sweep the sparsity knob. Our order parameter is **features-per-dimension**: how many features the network actually stores, divided by the number of dimensions. A value of 1 means "no superposition" (each stored feature owns a dimension); a value above 1 means superposition.

![Phase diagram](figures/02_phase_diagram.png)

The network sits in a **no-superposition phase** (features-per-dimension ≈ 1) in the dense world, then transitions into a **superposition phase** as the world gets sparse, packing several features into every dimension. This is structurally identical to a magnet ordering as temperature drops: an order parameter responding to a control parameter, with a recognisable transition region.

**Theory vs. measurement (right panel).** A simple argument predicts the *shape* of this curve. A feature is worth storing (in superposition) once its importance clears a threshold that shrinks as the world gets sparser — because the cost of overlapping two features is paid only when they fire *together*, a probability that falls like density². Since this model gives features geometrically decaying importance, the number that clear the threshold grows **linearly in log(1/density)**. The data follow that predicted form closely (R² ≈ 0.99). What the argument does *not* fix is the slope: it depends on O(1) constants (the exact collision penalty) that we do not compute, so the line is the predicted functional form with the prefactor measured, not a parameter-free fit. (The matching parameter-free statement is for the minimal *n=2, m=1* model, where the no-superposition → superposition boundary can be derived exactly to sit at the dense limit.)

### Experiment 3 — the geometry is quantized, and it solves a packing problem

`python experiments/03_feature_geometry.py`

Back to the n = 5, m = 2 model, sweeping sparsity finely.

![Feature geometry and packing](figures/03_feature_geometry.png)

**(A) Quantised geometry.** To quantify "how many dimensions a feature occupies" we use the **feature dimensionality** of Elhage et al. (with $\hat{W}_i = W_i / \lVert W_i \rVert$):

$$D_i = \frac{\lVert W_i \rVert^2}{\sum_{j=1}^{n} \bigl(\hat{W}_i \cdot W_j\bigr)^2}.$$

The numerator measures how strongly feature $i$ is represented; the denominator measures how much it shares its directions with all features (including itself). It satisfies $0 \le D_i$ and $\sum_i D_i \le m$ — the network has only $m$ dimensions to hand out. For $k$ unit vectors arranged as a **regular polygon** in 2D ($m=2$), every pairwise angle is a multiple of $2\pi/k$ and the denominator collapses to a clean value (derived in the Appendix), giving

$$D_i = \frac{2}{k}.$$

So the dimensionality does not drift smoothly — it **locks onto a discrete ladder** $2/k$, each value a specific shape:

| $D_i$ | $k$ | Shape the features form |
|---|---|---|
| $1$   | 1 | a feature alone on its own axis |
| $2/3$ | 3 | three features at 120° (a triangle) |
| $1/2$ | 4 | two features back-to-back (a pair) |
| $2/5$ | 5 | five features at 72° (a pentagon) |

The flat plateaus separated by jumps are the signature of distinct **geometric phases** — like discrete energy levels in a physical system. (The $2/3$ line is drawn for reference; this particular model jumps past it.)

**(B) A solved packing problem.** We define a **frustration energy** — the total squared overlap between *unit* feature directions, the network's analogue of electrostatic repulsion:

$$E(W) = \sum_{i<j} \bigl(\hat{W}_i \cdot \hat{W}_j\bigr)^2.$$

For $k$ unit vectors equally spaced on the circle, this has the closed form $E_k = \tfrac{1}{4}k(k-2)$ for $k \ge 3$ (derived in the Appendix). For the pentagon, $E_5 = \tfrac{1}{4}\cdot 5 \cdot 3 = 3.75$. Comparing the trained network to this ideal:

> **learned energy = 3.750, ideal regular-pentagon energy = 3.750** — they match to numerical precision.

The network, simply by minimizing reconstruction error, has independently found the minimum-energy packing — the same flavour of answer the Thomson problem gives for points repelling on a sphere.

---

## 6. Run it yourself

```bash
git clone <this-repo>
cd Superposition
pip install -e .            # installs torch, numpy, matplotlib
python experiments/01_superposition_emergence.py
python experiments/02_phase_diagram.py
python experiments/03_feature_geometry.py
pytest                      # fast sanity tests
```

Figures are written to [`figures/`](figures). Everything runs on CPU in a few minutes.

## 7. Repository layout

```
src/superposition/
  data.py      sparse synthetic features (the "world")
  model.py     the toy autoencoder
  train.py     training loop + config
  metrics.py   order parameters: dimensionality, frustration energy
  viz.py       plotting helpers
experiments/   three scripts, each producing one figure above
tests/         fast checks of the engine and metrics
```

## 8. Appendix — definitions and derivations

### A. The regular-polygon identities

For $k$ unit vectors equally spaced on the circle, write $\hat{W}_a = (\cos\theta_a, \sin\theta_a)$ with $\theta_a = 2\pi a / k$, so that $\hat{W}_a \cdot \hat{W}_b = \cos\!\big(\tfrac{2\pi (a-b)}{k}\big)$. The one fact we need is that for $k \ge 3$ the cross term vanishes and

$$\sum_{d=0}^{k-1} \cos^2\!\Big(\tfrac{2\pi d}{k}\Big) = \frac{k}{2}.$$

**Dimensionality** $D_i = 2/k$ (used in Section 5A). With unit vectors $\lVert W_i \rVert = 1$, the denominator of $D_i$ is exactly the sum above, so

$$\sum_{j} \bigl(\hat{W}_i \cdot W_j\bigr)^2 = \sum_{d=0}^{k-1} \cos^2\!\Big(\tfrac{2\pi d}{k}\Big) = \frac{k}{2} \quad\Longrightarrow\quad D_i = \frac{1}{k/2} = \frac{2}{k}.$$

**Frustration energy** $E_k = \tfrac14 k(k-2)$ (used in Section 5B). Summing over unordered pairs, with $k$ ordered pairs for each nonzero difference $d$,

$$E_k = \sum_{a<b} \cos^2\!\Big(\tfrac{2\pi (a-b)}{k}\Big) = \frac{k}{2}\sum_{d=1}^{k-1}\cos^2\!\Big(\tfrac{2\pi d}{k}\Big) = \frac{k}{2}\Big(\frac{k}{2} - 1\Big) = \frac{k(k-2)}{4}.$$

For the pentagon $E_5 = \tfrac14 \cdot 5 \cdot 3 = 3.75$ (an antipodal pair, $k=2$, gives $E = 1$ separately).

### B. The phase boundary (minimal model $n=2$, $m=1$)

One feature can always be stored perfectly in a single dimension; the question is whether a *second* feature is worth adding. Write $W = (w_1, w_2)$ so $h = w_1 x_1 + w_2 x_2$, and compare two strategies.

**Dedicate** ($W = (1, 0)$, $b = 0$): then $\hat{x}_1 = \mathrm{ReLU}(x_1) = x_1$ and $\hat{x}_2 = 0$, so only feature 2's error survives:

$$L_{\text{ded}} = I\,\mathbb{E}[x_2^2] = I\,p \int_0^1 u^2\,du = \frac{I p}{3}.$$

**Superpose antipodally** ($W = (1, -1)$, $b = 0$): then $h = x_1 - x_2$, giving $\hat{x}_1 = \mathrm{ReLU}(x_1 - x_2)$ and $\hat{x}_2 = \mathrm{ReLU}(x_2 - x_1)$. If only one feature is on, reconstruction is *exact* (the ReLU clips the rest); error appears **only when both fire** (probability $p^2$), and then each feature's error equals $\min(x_1, x_2)$. With $\mathbb{E}[\min(u,v)^2] = 2\int_0^1\!\int_0^v u^2\,du\,dv = \tfrac16$ over the unit square,

$$L_{\text{sup}} = I\,p^2 \cdot 2\,\mathbb{E}[\min(u,v)^2] = I\,p^2 \cdot 2 \cdot \tfrac16 = \frac{I p^2}{3}.$$

**Boundary.** Setting $L_{\text{sup}} = L_{\text{ded}}$ gives $\tfrac{I p^2}{3} = \tfrac{I p}{3}$, i.e. $p = 1$. For every $p < 1$ (any sparsity at all) we have $L_{\text{sup}} < L_{\text{ded}}$: **superposition wins, and the no-superposition phase survives only at the dense point $p = 1$.** The mechanism is the competing scaling

$$\underbrace{\text{benefit} \;\propto\; p}_{\text{feature is on}} \qquad \text{vs.} \qquad \underbrace{\text{interference cost} \;\propto\; p^2}_{\text{two features collide}},$$

so the cost-to-benefit ratio $\propto p \to 0$ as the world gets sparse.

### C. From the boundary to the log-linear count (Exp 2)

In the full model ($n > m$), adding feature $k$ in superposition inflicts interference on the already-stored, more important features. The marginal cost scales as $c\,p^2 \sum_{j<k} I_j$ against a marginal benefit $\tfrac{p}{3} I_k$, where $c = O(1)$ is an undetermined collision constant. Storing feature $k$ is worthwhile while $\tfrac{p}{3} I_k \gtrsim c\,p^2 \sum_{j<k} I_j$. With geometric importance $I_k = r^{\,k-1}$ and $\sum_{j<k} I_j \to \tfrac{1}{1-r}$,

$$r^{\,k-1} \;\gtrsim\; \frac{3 c\,p}{1-r} \quad\Longrightarrow\quad k \;\lesssim\; k_0 + \frac{\ln(1/p)}{\ln(1/r)}.$$

So **features-per-dimension $k/m$ grows linearly in $\ln(1/\text{density})$**, with slope $\sim \tfrac{1}{m \ln(1/r)}$ up to the constant $c$. Exp 2 confirms this *form* ($R^2 = 0.99$); the unknown $c$ is exactly why the measured slope ($0.79$ per e-fold) and this crude estimate ($1.95$) differ by an $O(1)$ factor.

## 9. References

- Elhage et al., **"Toy Models of Superposition"**, Anthropic / Transformer Circuits Thread, 2022. <https://transformer-circuits.pub/2022/toy_model/index.html>
- J. J. Thomson (1904) and the long line of work on minimum-energy point configurations on the sphere (the Thomson / Tammes problems).

---

*Built as a research demonstration. The model, the dimensionality metric, the quantized geometry, and the phase-change idea are all from the original paper — this repo reproduces them from scratch. My additions are the explicit packing-energy (Thomson-style) measurement and the order-parameter presentation of the phase diagram, which sharpen the paper's own framing rather than replace it.*
