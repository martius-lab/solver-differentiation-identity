# [Backpropagation through Combinatorial Algorithms: Identity with Projection Works](https://arxiv.org/abs/2205.15213) 
By [Subham S. Sahoo*](https://s-sahoo.github.io), [Anselm Paulus*](https://www.cs.cornell.edu/~kuleshov), [Marin Vlastelica](https://jimimvp.github.io), [Vít Musil](https://gimli.ms.mff.cuni.cz/~vejtek/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/), [Georg Martius](https://is.mpg.de/person/gmartius) 

Embedding discrete solvers as differentiable layers equips deep learning models with combinatorial expressivity and discrete reasoning. However, these solvers have zero or undefined derivatives, necessitating a meaningful gradient replacement for effective learning. Prior methods smooth or relax solvers, or interpolate loss landscapes, often requiring extra solver calls, hyperparameters, or sacrificing performance. We propose a principled approach leveraging the geometry of the discrete solution space, treating the solver as a negative identity on the backward pass, with theoretical justification. Experiments show our hyperparameter-free method competes with complex alternatives across tasks like discrete sampling, graph matching, and image retrieval. We also introduce a generic regularization that prevents cost collapse and enhances robustness, replacing previous margin-based approaches.

Link to the paper on [arXiv](https://arxiv.org/abs/2205.15213).

## Requirements
* `tensorflow==2.3.0` or `tensorflow-gpu==2.3.0`
* `numpy==1.18.5`
* `matplotlib==3.1.1`
* `scikit-learn==0.24.1`
* `tensorflow-probability==0.7.0`

[IdentitySubsetLayer](https://github.com/martius-lab/solver-differentiation-identity/blob/main/discrete-VAE-experiments-neurips-identity.ipynb)
class implements the backpropation through the sampling process of $k-hot$ vectors using our proposed identity method.

```
class IdentitySubsetLayer(tf.keras.layers.Layer):
       
    @tf.custom_gradient
    def identity_layer(self, logits_with_noise, hard=False):
        threshold = tf.expand_dims(tf.nn.top_k(logits_with_noise, self.k, sorted=True)[0][:,-1], -1)
        y_map = tf.cast(tf.greater_equal(logits_with_noise, threshold), tf.float32)

        def custom_grad(dy):
            return dy, hard

        return y_map, custom_grad

    def call(self, logits, hard=False):
        logits = logits + sample_gumbel_k(tf.shape(logits))
        
        # Project onto the hyperplane
        logits = logits - tf.reduce_mean(logits, 1)[:, None]
        
        # Project onto the sphere
        logits = logits / tf.norm(logits, axis=1)[:, None]
        
        return self.identity_layer(logits, hard)
```

<sup>This notebook was adapted from [here](https://github.com/nec-research/tf-imle).</sup>
