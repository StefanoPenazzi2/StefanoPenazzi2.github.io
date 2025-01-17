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

$$

L(\pi, f, g) = \int_{\Chi \times \Chi} \|x - y\|^2 \pi(x, y) \, dy \, dx 
+ \int_{\Chi} \left( p(x) - \int_{\Chi} \pi(x, y) \, dy \right) f(x) \, dx 
+ \int_{\Chi} \left( q(y) - \int_{\Chi} \pi(x, y) \, dx \right) g(y) \, dy.

$$


$$

L(\pi, f, g) = \mathbb{E}_{x \sim p} \left[ f(x) \right] + \mathbb{E}_{y \sim q} \left[ g(y) \right] 
+ \int_{\mathcal{X} \times \mathcal{X}} \left( \|x - y\|_2 - f(x) - g(y) \right) \pi(x, y) \, dy \, dx.

$$

