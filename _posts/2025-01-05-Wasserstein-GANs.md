## Wasserstein Generative Adversarial Neural Networks

---

The Wasserstein distance seeks to minimize the objective 

$$

\arg\min_{\theta} W(p, p_\theta) = \arg\min_{\theta} \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right].

$$

blablabla

### Kantorovich-Rubinstein Duality

**Theorem**

$$

W(p, q) = \inf_{\pi \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \pi} \left[ \|x - y\|_2 \right] = \sup_{\|h\|_L \leq 1} \left[ \mathbb{E}_{x \sim p} [h(x)] - \mathbb{E}_{y \sim q} [h(x)] \right]

$$


$$

L(\pi, f, g) = \int_{\mathcal{X} \times \mathcal{X}} \|x - y\|^2 \pi(x, y) \, dy \, dx 
+ \int_{\mathcal{X}} \left( p(x) - \int_{\mathcal{X}} \pi(x, y) \, dy \right) f(x) \, dx 
+ \int_{\mathcal{X}} \left( q(y) - \int_{\mathcal{X}} \pi(x, y) \, dx \right) g(y) \, dy.

$$


