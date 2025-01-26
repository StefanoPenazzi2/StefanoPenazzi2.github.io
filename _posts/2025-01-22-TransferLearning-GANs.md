## Transfer Learning in GANs

---

***Fine-tuning***

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/gans_transferlearning/finetuning.png?raw=true)

***L2-SP***

***Adaptive Batch Normalization (Domain Adaptation)***

![alt text](https://github.com/StefanoPenazzi2/StefanoPenazzi2.github.io/blob/main/imgs/gans_transferlearning/abn.png?raw=true)

- For neuron $j$ in DNN:
   - Concatenate neuron responses on all input of target domain $t$: $\mathbf{x}_j = [\ldots, x_j(m), \ldots]$
   - Compute the mean and variance of the target domain:
   $$ 
   \mu_j^t = \mathbb{E}(x_j^t), \quad \sigma_j^t = \sqrt{\text{Var}(x_j^t)}
   $$

- For neuron $j$ in DNN, testing input $m$ in target domain:
   - Compute BN output 
   $$
   y_j(m) := \gamma_j \frac{x_j(m) - \mu_j^t}{\sigma_j^t} + \beta_j
   $$



***Generative latent optimization (GLO)***

***MineGAN***

***FreezeD***

***Feature distillation***






