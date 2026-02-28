# AdamWClip: AdamW with adaptive gradient clipping

AdamWClip is an optimizer that extends AdamW with adaptive gradient clipping. 
It automatically adapts the gradient clipping thresholds to the gradient statistics of each parameter resulting in equivariant thresholds with respect to scaling the gradients. 
This makes finding suitable clipping thresholds much easier (usually, the default threshold of AdamWClip is good to go). 
Furthermore, by directly utilizing the internal state variables of Adam, AdamWClip doesn't require additional memory (and only a marginal computational overhead). 

## Useage

To use AdamWClip in your pytorch project, simply run the following:

```python
%pip install AdamWClip
from AdamWClip import AdamWClip
...
optimizer = AdamWClip(model.parameters(),*args)
```

On top of the standard parameters from AdamW, AdamWClip offers the following additional parameters:  

- clip_grad_adapt: adaptive gradient clipping threshold in terms of standard deviations of the clipped gradient distribution. If set to None, this optimizer behaves exactly like AdamW (default: 3)
- clip_grad_min: minimum value for the adaptive gradient clipping threshold (default: 1e-10)
- clip_grad_warm_up: (Optional) number of initial update steps without gradient clipping to obtain more reasonable gradient statistics at the beginning (default: 0)

In most instances, the default values should be fine.



If this optimizer becomes useful to you, please consider citing this repository :)
