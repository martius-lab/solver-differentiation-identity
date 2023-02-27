# solver-differentiation-identity
Repository for the code of our ICLR paper: [Backpropagation through Combinatorial Algorithms: Identity with Projection Works](https://arxiv.org/abs/2205.15213) 

## Requirements
* tensorflow==2.3.0 or tensorflow-gpu==2.3.0
* numpy==1.18.5
* matplotlib==3.1.1
* scikit-learn==0.24.1
* tensorflow-probability==0.7.0

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
