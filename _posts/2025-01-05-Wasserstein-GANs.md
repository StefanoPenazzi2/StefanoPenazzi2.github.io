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

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/wasserstein_dual_transport_plan.png?raw=true)



### Kantorovich-Rubinstein Duality

Kantorovich-Rubinstein Duality reformulate $W(p,q)$ as the solution to a maximization over 1-Lipschitz functions 

**Theorem**

$$

W(p, q) = \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right] = \sup_{\|h\|_L \leq 1} \left[ \mathbb{E}_{x \sim p} [h(x)] - \mathbb{E}_{y \sim q} [h(x)] \right]

$$

The joint probability distribution $\pi(x, y)$ is introduced as a computational tool rather than representing an
inherent physical or probabilistic relationship between $x$ and $y$. The use of $\pi(x, y)$ facilitates the
modeling of transport plans that minimize the expected transportation cost between $p(x)$ and $q(y)$.
While $\pi(x, y)$ satisfies real constraints, ensuring correct marginals of $p(x)$ and $q(y)$, it primarily 
serves to bridge the source and target distributions in an abstract space. The interpretation of $\pi(x, y)$ as a
joint distribution is metaphorical; its practical significance lies in encapsulating all possible ways to map the
source distribution to the target within the given constraints.

**Proof 1**


$W(p, q) = \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right]$ this can be rewritten into

$$
\begin{aligned}
\min_{\pi} \quad & \int_{\chi \times \chi} \|x - y\|^2 \pi(x, y) \, dy \, dx \\
\textrm{s.t.} \quad & \int_{\chi} \pi(x, y) \, dy = p(x)\\
  &   \int_{\chi} \pi(x, y) \, dx =  q(y)\\
\end{aligned}
$$

The objective is to minimize the expected cost of transporting a distribution $p(x)$ to $q(y)$,
where the cost is quantified by the squared Euclidean distance $\|x - y\|^2$ weighted by the
joint distribution $\pi(x, y)$. The constraints ensure that the marginals of $\(\pi\)$ match
the given distributions $p(x)$ and $q(y)$.

By discretizing the supports of $x$ and $y$, we can replace the continuous integrals with finite sums,
which simplifies the problem into a discrete optimization framework.
This transformation allows us to use linear programming, where the discrete set of values for $x$ and $y$
results in specific constraints for each pair $(x_i, y_j)$. The discrete formulation eases computation
and aligns with practical scenarios where data is often collected or represented in discrete form.

$$
\begin{aligned}
\min_{\pi} \quad & \sum_{i=1}^{n}\sum_{j=1}^{m} \|x_i - y_j\|^2 \pi(x_i, y_j) \\
\textrm{s.t.} \quad & \sum_{j=1}^{n} \pi(x_i, y_j) = p(x_i) \; i = 1, \ldots, n.\\
  &   \sum_{i=1}^{n} \pi(x_i, y_j) = q(y_j) \; j = 1, \ldots, m.\\
\end{aligned}
$$

The strong duality theorem states that if either the primal or dual linear program has an optimal solution,
the other also attains an optimal solution, with both achieving the same objective value, thereby making
the weak duality bounds tight (the result is closely related to Farkas' Lemma, which provides a foundational
condition for the solvability of linear inequalities).

$$

\begin{aligned}
& \min_{k} \, c^T k & = & \quad \max_{z} \, b^T z \\
& \text{s.t. } Ak \geq b, &  & \quad \text{s.t. } A^T z \leq c, \\
& \quad k \geq 0, &  & \quad z \geq 0.
\end{aligned}

$$

In order to transform our discretized constrained optimization problem in the primal standard form above a few transformations are required.
The image illustrates the transformation of the discretized constrained optimization problem into its necessary primal form.
This process involves several key transformations: 

- The joint distribution $\pi$ is vectorized into $k$, which serves as the model's variable.
- The marginals' probabilities are concatenated into vector $b$.
- The distance matrix is vectorized into $c$.
- The matrix $A$ is constructed to ensure that given $b$ and $k$, the constraints on the marginals are correctly enforced.

These transformations facilitate the handling of the optimization problem in a structured and solvable form.

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/wasserstein_dual_lin_prog.png?raw=true)

In our optimization model, each dual variable is associated with a specific primal constraint. We have two types of constraints:
those ensuring the marginal $p(x)$ is respected, and those ensuring the marginal $q(y)$ is respected. For each discrete value $x_i$
and $y_j$, there is a corresponding constraint. The dual variables linked to the $p(x)$ constraints are denoted by $f(x)$,
while those linked to the $q(y)$ constraints are denoted by $g(y)$. While treating these dual variables as functions is not strictly
necessary for solving the dual problem, it provides a strategic advantage.
The dual problem can be rewritten as below

$$

\begin{align*}
\max_{f, g} \quad & \sum_{i=1}^{n} f(x_i) p(x_i) + \sum_{j=1}^{m} g(y_j) q(y_j) \\
\text{s.t.} \quad & f(x_i) + g(y_j) \leq \|x_i - y_j\|^2, \quad \forall i, j.
\end{align*}

$$

Let's examine the dual constraints more closely. The minimum value of the distance $\|x_i - y_j\|^2$ is 0 when $x_i = y_j$,
which results in the strictest constraint $f(x_i) + g(x_i) \leq 0$. This implies that $f(x_i)$ and $g(x_i)$ cannot both
be positive at the same time, enforcing $f(x_i) \leq -g(x_i)$. The maximum value of $f(x_i) + g(x_i)$ that respects the
constraints is 0. We can increase the value of $f(x_i)$ indefinitely, provided $f(x_i) = -g(x_i)$. Since the objective function of the dual problem is

$$
b^T z = p(x_1)f(x_1) + p(x_2)f(x_2) + \ldots + q(y_1)g(y_1) + q(y_2)g(y_2) + \ldots
$$

$f(x_i)$ always impacts the objective function positively, while $-g(x_i)$ impacts it negatively. Therefore,
the maximum value is achieved when the constraint $f(x_i) \leq -g(x_i)$ is an equality. This means we can substitute $g(x_i) = -f(x_i)$.


$$

\begin{align*}
W(p, q) = & \sup_{\substack{f, g \\ f(x) + g(y) \leq \|x - y\|_2}} \left[ \mathbb{E}_{x \sim p} [f(x)] + \mathbb{E}_{y \sim q} [g(y)] \right] \\
\leq & \sup_{\|h\|_L \leq 1} \left[ \mathbb{E}_{x \sim p} [h(x)] - \mathbb{E}_{y \sim q} [h(y)] \right] \leq W(p, q).
\end{align*}

$$



**Proof 2**
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