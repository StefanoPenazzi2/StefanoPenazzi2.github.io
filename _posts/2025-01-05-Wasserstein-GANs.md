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
Adding the Lagrangian multipliers $f, g : \mathcal{X} \rightarrow \mathbb{R}$.

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

L(\pi, f, g) = \mathbb{E}_{x \sim p} \left[ f(x) \right] + \mathbb{E}_{y \sim q} \left[ g(y) \right] 
+ \int_{\mathcal{X} \times \mathcal{X}} \left( \|x - y\|_2 - f(x) - g(y) \right) \pi(x, y) \, dy \, dx.

$$

Since this function satisfies the conditions for strong duality, we can express it as:

$$

W(p, q) = \inf_{\pi} \sup_{f, g} L(\pi, f, g) = \sup_{f, g} \inf_{\pi} L(\pi, f, g).

$$

