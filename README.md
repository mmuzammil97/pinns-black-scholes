# Physics-Informed Neural Networks for Black-Scholes Option Pricing

This project demonstrates how to use Physics-Informed Neural Networks (PINNs) to solve differential equations, culminating in pricing a European call option from the Black-Scholes PDE.  
Before the option-pricing problem, the notebook builds intuition on two benchmark problems:

1. A 1D damped harmonic oscillator (ODE)
2. The 1D heat equation (PDE)

These examples show how PINNs encode physics directly into the training objective and how that differs from purely data-driven neural networks.

## What Is a PINN?

A PINN is a neural network trained not only on observed/known data but also on the governing differential equation.

Given a differential operator:

$$
\mathcal{N}[u](x,t) = 0
$$

the network $u_\theta(x,t)$ is trained to minimize:

1. **Physics loss**: residual of the PDE/ODE at collocation points in the domain  
2. **Constraint loss**: initial conditions, boundary conditions, and/or terminal conditions  
3. **(Optional) data loss**: supervised mismatch to known data

So the model is constrained by physical law during training, which often improves generalization when labeled data are limited.

## Why PINNs for Finance?

Option pricing under Black-Scholes is governed by a PDE. PINNs are a natural fit because:

1. The PDE and financial boundary/terminal constraints are known exactly.
2. We can sample collocation points anywhere in $(S,t)$ space without requiring labeled market data at every point.
3. We can recover a full price surface $V(S,t)$, not just point estimates.

## Notebook Walkthrough

The notebook `PINNs_for_BlackScholes.ipynb` is organized into four conceptual blocks.

### 1) Network Building Blocks

You first define a fully-connected feed-forward neural network (FCNN) using `torch.nn.Sequential` with `tanh` activations.  
This architecture is reused across examples, with only input/output dimensions changing:

1. Oscillator: input $t$, output $u(t)$
2. Heat equation: input $(x,t)$, output $u(x,t)$
3. Black-Scholes: input $(S,t)$, output $V(S,t)$

### 2) Damped Harmonic Oscillator (ODE)

The notebook solves:

$$
u''(t) + cu'(t) + ku(t) = 0
$$

with initial conditions $u(0)=u_0$, $u'(0)=v_0$.

#### PINN setup for oscillator

At collocation points $t_f$, the network produces $u_\theta(t_f)$.  
Using autograd:

1. $u_t = \frac{du_\theta}{dt}$
2. $u_{tt} = \frac{d^2u_\theta}{dt^2}$

Residual:

$$
f(t) = u_{tt} + cu_t + ku
$$

Loss combines:

1. `MSE(f, 0)` (physics)
2. `MSE(u(0), u0)` (IC displacement)
3. `MSE(u_t(0), v0)` (IC velocity)

This section also generates training GIFs that compare analytic and PINN trajectories over epochs.

#### Supervised NN baseline

A traditional neural network is trained on sampled $(t, u(t))$ pairs only (no physics residual).  
This gives a direct comparison: PINN constraints vs purely supervised fitting.

### 3) Heat Equation (PDE)

The notebook then solves:

$$
u_t = \alpha u_{xx}, \quad x \in [0,1], \; t \in [0,T]
$$

with:

1. Initial condition $u(x,0)=\sin(\pi x)$
2. Boundary conditions $u(0,t)=u(1,t)=0$

PINN residual:

$$
f(x,t)=u_t-\alpha u_{xx}
$$

Loss terms:

1. PDE residual loss over interior collocation points
2. Initial-condition loss
3. Boundary-condition loss

This block includes:

1. Line profile comparisons to exact solution
2. Error reporting (L2-type metric)
3. Spatiotemporal GIF visualizations and heatmaps

### 4) Black-Scholes Equation (Main Goal)

For a European call option $V(S,t)$, the Black-Scholes PDE is:

$$
V_t + \frac{1}{2}\sigma^2 S^2 V_{SS} + rS V_S - rV = 0
$$

with:

1. **Terminal condition** at $t=T$:  
   $$
   V(S,T)=\max(S-K,0)
   $$
2. **Boundary conditions**:
   $$
   V(0,t)=0,\quad
   V(S_{\max},t)\approx S_{\max}-K e^{-r(T-t)}
   $$

#### Black-Scholes PINN loss

The notebook trains a network $V_\theta(S,t)$ with:

1. PDE residual loss in the interior of $(S,t)$
2. Terminal payoff loss
3. Boundary condition losses at $S=0$ and $S=S_{\max}$

An analytic Black-Scholes formula (`scipy.stats.norm`) is used as a reference for validation/plots.

#### Outputs in this section

1. Price curves $V(S,t)$ at multiple time slices
2. PINN vs analytic comparisons
3. 3D option surface visualizations over stock price and time

## Generated Media and GitHub Rendering

The notebook generates GIF files and now includes:

1. Inline display calls after GIF creation (`IPython.display.Image`)
2. A markdown section embedding the GIF assets by relative path

This helps GitHub notebook previews show animations/images inline as long as the GIF files are committed.

## Practical Notes

1. Training is computationally expensive in several sections (large epoch counts).
2. Results vary run-to-run due to random sampling unless random seeds are fixed.
3. For faster iteration, temporarily reduce epochs and collocation counts.
4. For publication-quality results, use the larger default training settings.

## Dependencies

Install from:

`requirements.txt`

The project currently expects:

1. `numpy`
2. `matplotlib`
3. `torch`
4. `imageio`
5. `scipy`
