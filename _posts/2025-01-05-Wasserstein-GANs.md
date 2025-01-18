## Wasserstein Generative Adversarial Neural Networks

---

Let $p$ and $q$ be two probability distributions on $\mathbb{R}^n$.  $\Pi(p, q)$ is the set of all the probability distributions $\pi(p, q)$ on 
$\mathbb{R}^n \times \mathbb{R}^n$ such that the marginals of $\pi$ are $p(x)$ and $q(y)$. The Wasserstein-1 distance is

$$

W(p, q) = \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right].

$$

The Wasserstein distance $W(p,q)$ intuitively represents the minimum amount of work required to transport mass from one distribution
$p$ to another distribution $q$. When two distributions are very similar, the mass transferred from one to the other is largely allocated to points in
$x$ that are very close to corresponding points in $y$, ensuring an efficient transformation of $p$ into $q$.
However, directly computing $W(p,q)$ is often intractable due to the sheer size of the set  $\Pi(p, q)$


### Kantorovich-Rubinstein Duality

Kantorovich-Rubinstein Duality reformulate $W(p,q)$ as the solution to a maximization over 1-Lipschitz functions 

**Theorem**

$$

W(p, q) = \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right] = \sup_{\|h\|_L \leq 1} \left[ \mathbb{E}_{x \sim p} [h(x)] - \mathbb{E}_{y \sim q} [h(x)] \right]

$$

**Proof**
Adding the Lagrangian multipliers $f, g : \chi \rightarrow \mathbb{R}$.

$$
\begin{split}
L(\pi, f, g) = \int_{\chi \times \chi} \|x - y\|^2 \pi(x, y) \, dy \, dx 
+ \int_{\chi} \left( p(x) - \int_{\chi} \pi(x, y) \, dy \right) f(x) \, dx \\
+ \int_{\chi} \left( q(y) - \int_{\chi} \pi(x, y) \, dx \right) g(y) \, dy.
\end{split}
$$

Lagrange multipliers are employed to transform a constrained optimization problem into an unconstrained one while preserving the same optimal solution.
In this context, the constraints ensure that the marginals of $ \pi(\cdot, \cdot)$ remain $p$ and $q$. 
For each value in the support of $p$, a penalty term $p(x) - \int_{\chi} \pi(x, y) \, dy$ is introduced 
and scaled by its corresponding Lagrange multiplier $f(x)$.
Minimizing the objective function thus involves reducing the discrepancy between $p(x)$ and $\int_{\chi} \pi(x, y) \, dy$
ensuring the marginal condition is satisfied.

Collecting terms algebraically, we can rewrite the Lagrangian as

$$
\begin{split}
L(\pi, f, g) = \mathbb{E}_{x \sim p} \left[ f(x) \right] + \mathbb{E}_{y \sim q} \left[ g(y) \right]
+ \int_{\chi \times \chi} \left( \|x - y\|_2 - f(x) - g(y) \right) \pi(x, y) \, dy \, dx.
\end{split}
$$

Since this function satisfies the conditions for strong duality, we can express it as:

$$

W(p, q) = \inf_{\pi} \sup_{f, g} L(\pi, f, g) = \sup_{f, g} \inf_{\pi} L(\pi, f, g).

$$

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/wasserstein_dual_transport_plan.png?raw=true)


$$
\begin{aligned}
\min_{\pi} \quad & \int_{\chi \times \chi} \|x - y\|^2 \pi(x, y) \, dy \, dx \\
\textrm{s.t.} \quad & \int_{\chi} \pi(x, y) \, dy = p(x)\\
  &   \int_{\chi} \pi(x, y) \, dx =  q(y)\\
\end{aligned}
$$


$$
\begin{aligned}
\min_{\pi} \quad & \sum_{i=1}^{n}\sum_{j=1}^{m} \|x_i - y_j\|^2 \pi(x_i, y_j) \\
\textrm{s.t.} \quad & \sum_{j=1}^{n} \pi(x_i, y_j) = p(x_i) \; i = 1, \ldots, n.\\
  &   \sum_{i=1}^{n} \pi(x_i, y_j) = q(y_j) \; j = 1, \ldots, m.\\
\end{aligned}
$$

$$

\begin{aligned}
& \min_{x} \, c^T x & & \quad \max_{y} \, b^T y \\
& \text{s.t. } Ax \geq b, & & \quad \text{s.t. } A^T y \leq c, \\
& \quad x \geq 0, & & \quad y \geq 0.
\end{aligned}

$$


$$

\begin{align*}
\max_{f, g} \quad & \sum_{i=1}^{n} f(x_i) p(x_i) + \sum_{j=1}^{m} g(y_j) q(y_j) \\
\text{s.t.} \quad & f(x_i) + g(y_j) \leq \|x_i - y_j\|^2, \quad \forall i, j.
\end{align*}

$$