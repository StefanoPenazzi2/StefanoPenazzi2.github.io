## F-divergence duality

---

### Convex conjugate

**Convex conjugate** _Let $ f : I \to \mathbb{R} $ be a convex function, where $ I \subseteq \mathbb{R} $ is an interval. The convex conjugate of $ f $ is another function $ f^* : I^* \to \mathbb{R} $ defined as:_

$$
f^*(y) = \sup_{x \in \mathbb{R}^n} \left\{ \langle y, x \rangle - f(x) \right\},
$$

_where $ I^* $ is the domain of $ f^* $, determined by the values of $ y $ for which the supremum is finite._ 

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/convex_conjugate.png?raw=true)

**Properties** _The convex conjugate $ f^* $ of $ f $ satisfies:

- $ f^* $ _is continuous on its domain,_
- $ f^* $ _is convex,_
- Biconjugation: $$f^{**} = f$$

### F-divergences

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/2d_dist_comp_gans.png?raw=true)

Let P and Q be two probability distributions over a space $ \Omega $, such that $ P \ll Q $, that is, P is
absolutely continuous with respect to Q (i.e., $ P(A) = 0 $ whenever $ Q(A) = 0 $). Then, for a convex function $ f: [0, +\infty) \to (-\infty,+\infty] $
such that $ f(x) $ is finite for all $ x > 0 $, $ f(1) = 0 $, and $ f(0) = \lim_{t \to 0^+} f(t) $
(which could be infinite), the f-divergence of P from Q is defined as.

$$
D_f(P \| Q) = 
\begin{cases} 
\int_\Omega f\left(\frac{dP}{dQ}\right) \, dQ, & \text{if } P \ll Q, \\ 
+\infty, & \text{otherwise},
\end{cases}
$$

We call $ f $ the generator of $ D_f $.

In real applications, there is usually a reference distribution $ \mu $ on $ \Omega $ (for example, when $ \Omega = \mathbb{R}^n $,
the reference distribution is the Lebesgue measure), such that $ P, Q \ll \mu $. Then we can use the Radon–Nikodym theorem to take
their probability densities $ p $ and $ q $, giving

$$

D_f(P \parallel Q) = \int_{\Omega} f\left(\frac{p(x)}{q(x)}\right) q(x) \, d\mu(x)

$$

The Lebesgue measure $\mu(x)$ is a generalization of the concept of length or volume in mathematical analysis.
It allows integration over more general sets in a space $\Omega$ than traditional Riemann integration.
When the underlying space $\Omega$ is the real line or $\mathbb{R}^n$, and $ p(x) $ and $ q(x) $ are probability density
functions with respect to the Lebesgue measure, then $ d\mu(x) $ can often be written as $ dx $.
This is because the Lebesgue measure on $\mathbb{R}$ or $\mathbb{R}^n$ simply corresponds to the usual way we measure length
or volume on those spaces.

$$

D_f(P \parallel Q)  = \int_{\Omega} f\left(\frac{p(x)}{q(x)}\right) q(x) \, d(x)

$$

For the discrete case, with $Q(x)$ and $P(x)$ being the respective pmfs, we can also write

$$

D_f(P \parallel Q) \triangleq \mathbb{E}_Q \left[ f \left( \frac{dP}{dQ} \right) \right] = \sum_x Q(x) f \left( \frac{P(x)}{Q(x)} \right)

$$

### F-divergences duality

**Theorem 1** _For any f-divergence, we have:_

$$
D_f(P \| Q) = \sup_{g : \Omega \to \mathbb{R}} \left\{ \mathbb{E}_P[g(X)] - \mathbb{E}_Q[f^*(g(X))] \right\},
$$

where:

- $ P $ and $ Q $ are probability distributions over a common measurable space $ (\Omega, \mathcal{F}) $,
- $ f : I \to \mathbb{R} $ is a convex function,
- $ f^* $ is the convex conjugate of $ f $,
- $ g(X) $ can be any function for which $ \mathbb{E}_Q[f^*(g(X))] $ and $ \mathbb{E}_P[g(X)] $ are finite,
- $ \mathbb{E}_Q[f^{*}(g(X))] $ is the expectation under $ Q $, 
- $ \mathbb{E}_P[g(X)] $ is the expectation of $ g(X) $ under $ P $



**Proof**

$$
\begin{align}
D_f(P \| Q) = \int_\chi  f\left( \frac{dP}{dQ} (x) \right) dQ(x) \\
= \int_\chi \sup_{y \in dom(f^*)} \left( y\frac{dP}{dQ} (x) - f^*(y) \right) dQ(x) \\
\ge \int_\chi \left( g(x)\frac{dP}{dQ} (x) - f^*(g(x)) \right) dQ(x) \\
= \int_\chi g(x)\frac{dP}{dQ} (x) dQ(x) - \int_\chi f^*(g(x)) dQ(x) \\
= \int_\chi g(x) dP(x) - \int_\chi f^*(g(x)) dQ(x) \\
= \mathbb{E}_P[g(X)] - \mathbb{E}_Q[f^*(g(X))]
\end{align}
$$

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/g_x_no_opt.png?raw=true)

It can be shown that the above lower bound is tight and achieved by $ g(x) =  f^{'}(\frac{dP}{dQ} (x))$

**Example**

Given two normal distributions (Figure 1) with parameters:

- $ \mu_1 = 1 , \sigma_1 = 1 $ for the first distribution
- $ \mu_2 = 2 , \sigma_2 = 2 $ for the second distribution

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/1d_dist_comp_gans.png?raw=true)

we want to calculate the Kullback-Leibler (KL) divergence. As we know, KL divergence is not symmetric, and we have the following results:

- $ D_{KL}(P \| Q) = 0.441796 $
- $ D_{KL}(Q \| P) = 1.279359 $

Now, let's assume we are only able to sample from these two distributions and do not have access to their explicit forms.
This is where the concept of f-divergence duality comes into play.

In this scenario, we sample 10,000 points from both distributions. We use multiple arbitrary functions g(x) to assess the best one.
It is important to note that g(x) should be sufficiently flexible to approximate the unknown $ f^{'}(\frac{dP}{dQ} (x)) $ accurately.
For each chosen function g(x), we estimate its parameters by maximizing Equation 5 using the Nelder-Mead simplex algorithm. 
This approach allows us to estimate the divergence without needing explicit knowledge of the distributions themselves.

The benchmark is $ D_{KL}(Q \| P) = 1.279359 $. The closer the value obtained from the optimization is to this benchmark, the better the result.

In this first case, we use a linear function $g(x) = a + bx$. The algorithm is initialized with $ a = 0, b = 0$.
The figure below illustrates the values of the objective function and the corresponding parameters at each iteration of the optimization algorithm.
Lighter colors represent more recent iterations.
![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/g_x_linear_opt.png?raw=true)

The maximum value obtained by the algorithm is 1.529 at $ a = 1.567 $ and $ b = -1.039$

In this second case, we use a polynomial function $g(x) = a + bx + cx^2 + dx^3$. The algorithm is initialized with $ a = 0, b = 0, c = 0, d = 0$
![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/g_x_poli3_opt.png?raw=true)

The maximum value obtained by the algorithm is 3.040 at $ a = -1.444e-01,  b = 8.323e-01,  c = 4.859e-01,  d = -2.371e-01$

In the last case, we use the optimal solution $g(x) = \frac{df}{dx}(\frac{dP}{dQ} (x)) = -\frac{\phi(\mu_1,\sigma_1)}{\phi(\mu_2,\sigma_2)}$.
This would not be possible since the explicit form of the distributions is unknown. It is worth noting that the optimization algorithm used does
not inherently handle constraints. However, since constraints are necessary in this case (e.g., standard deviations must be non-negative),
the objective function incorporates these constraints using Lagrange multipliers. 
The algorithm is initialized with $ \mu_1 = 0, \sigma_1 = 1, \mu_2 = 0, \sigma_2 = 1$.
The figure below shows on the right the values of the objective function and the corresponding $ \mu_1,\sigma_1 $ at each iteration of the optimization algorithm and on the left 
the values of the objective function and the corresponding $ \mu_2,\sigma_2 $ at each iteration of the optimization algorithm.
![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/g_x_deriv_f_x_opt.png?raw=true)

The maximum value obtained by the algorithm is 1.435 at $ \mu_1 = 1.016e+00, \sigma_1 = 9.584e-01, \mu_2 = 2.378e+00, \sigma_2 = 2.332e+00$


**Conclusions**
In conclusion, f-divergence duality is a powerful approach for estimating divergence measures when the explicit forms of distributions
are unknown but sampling is feasible. It provides a principled framework that leverages optimization and flexible functions, such as 
g(x), to approximate the required quantities. While simple functions were used here to approximate $ \frac{df}{dx}(\frac{dP}{dQ} (x)) $, a neural network could be employed instead,
taking advantage of the Universal Approximation Theorem (UAT). This approach not only facilitates the practical computation of complex statistical
quantities but also underscores the importance of robust optimization techniques to achieve accurate and reliable results.