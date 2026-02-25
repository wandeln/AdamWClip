# AdamWClip: AdamW with adaptive gradient clipping

AdamWClip is an optimizer that extends AdamW with adaptive gradient clipping. 
This way, the gradient clipping thresholds automatically adapt to the gradient statistics and become equivariant with respect to scaling the gradients.
This makes finding suitable clipping thresholds much easier (usually, the default thresholds are good to go).
Furthermore, by directly utilizing the internal state variables of Adam, AdamWClip implements adaptive gradient clipping without an additional memory footprint.

## Useage

To use AdamWClip in your pytorch project, simply run the following:

```python
%pip install AdamWClip
from AdamWClip import AdamWClip
...
optimizer = AdamWClip(model.parameters(),*args)
```

On top of the standard parameters from AdamW, AdamWClip offers the following additional parameters:  
	
	- clip_grad_adapt: adaptive gradient clipping threshold in terms of standard deviations of the clipped gradient distribution (default: 3)
	- clip_grad_min: minimum value for the adaptive gradient clipping threshold (default: 0.01)
	- clip_grad_warm_up: Number of initial update steps without gradient clipping to obtain reasonable gradient statistics at the beginning (default: 10)

In most instances, the default values should be fine.

If this optimizer becomes useful to you, please consider citing this repository :)
