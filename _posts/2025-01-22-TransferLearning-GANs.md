## Transfer Learning in GANs

---

***Fine-tuning***

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/gans_transferlearning/finetuning.png?raw=true)

***L2-SP***

***Adaptive Batch Normalization (Domain Adaptation)***

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/gans_transferlearning/abn.png?raw=true)

Given the pre-trained DNN model and a target domain, Adaptive Batch Normalization algorithm is as follow:

- For neuron $j$ in DNN:
   - Concatenate neuron responses on all input of target domain $t$: $\mathbf{x}_j = [\ldots, x_j(m), \ldots]$
   - Compute the mean and variance of the target domain:
   $$ 
   \mu_j^t = \mathbb{E}(x_j^t) 
   $$
  
   $$
   \sigma_j^t = \sqrt{\text{Var}(x_j^t)}
   $$

- For neuron $j$ in DNN, testing input $m$ in target domain:
   - Compute BN output 
   $$
   y_j(m) := \gamma_j \frac{x_j(m) - \mu_j^t}{\sigma_j^t} + \beta_j
   $$



***Generative latent optimization (GLO)***

***MineGAN***

We aim to approximate a target real data distribution $p_T^{\text{data}}(x)$, derived from a set of real images $D_T$.
Given a critic $D$ and a generator $G$ that have been trained to approximate a source data distribution $p^{\text{data}}(x)$
through the generative distribution $p_g(x)$, the mining operation endeavors to learn a new generative distribution $p_T^g(x)$.
This is achieved by identifying regions within $p_g(x)$ that best approximate the target distribution $p_T^{\text{data}}(x)$,
while keeping $G$ fixed.
In order to find such regions, mining actually finds a new prior distribution $p^T_z(z)$ such that samples $G(z)$ with $z \sim p^T_z(z)$
are similar to samples from p^T_{data}(x).

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/gans_transferlearning/minegan.png?raw=true)


***FreezeD***

***Feature distillation***






