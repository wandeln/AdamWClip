import torch
from torch.optim.optimizer import Optimizer

class AdamWClip(Optimizer):
	"""
	AdamW optimizer with adaptive gradient clipping
	"""

	def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, clip_grad_adapt=3, clip_grad_min=1e-10, clip_grad_warm_up=0):
		"""
		params, lr, betas, eps, weight_decay are identical to AdamW optimizer
		On top of that, we have the following additional parameters:
		:clip_grad_adapt:   adaptive gradient clipping threshold in terms of standard deviations of the clipped gradient distribution. If set to None, this optimizer behaves exactly like AdamW (default: 3)
		:clip_grad_min:     minimum value for the adaptive gradient clipping threshold (default: 0.01)
		:clip_grad_warm_up: Number of initial update steps without gradient clipping to obtain reasonable gradient statistics at the beginning (default: 10)
		"""
		beta_1, beta_2 = betas
		defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay, clip_grad_adapt=clip_grad_adapt, clip_grad_min=clip_grad_min)
		self.iteration = 0
		self.clip_grad_warm_up = clip_grad_warm_up
		self.eps = eps
		super(AdamWClip, self).__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure=None, end=False):
		"""Performs a single optimization step."""
		loss = None
		if closure is not None:
			loss = closure()

		self.iteration += 1
		t = self.iteration

		for group in self.param_groups:
			lr = group['lr']
			beta_1 = group['beta_1']
			beta_2 = group['beta_2']
			weight_decay = group['weight_decay']
			clip_grad_adapt = group['clip_grad_adapt']
			clip_grad_min = group['clip_grad_min']
			
			lr_bias_correction1 = -lr/(1.0 - beta_1**t)
			bias_correction2 = 1.0/(1.0 - beta_2**t)
			
			thetas = []
			grads = []
			sum_grads = []
			sum_grad_grads = []

			for p in group['params']:
				if p.grad is None:
					continue
				
				thetas.append(p)
				grads.append(p.grad)
				
				# Momentum part
				param_state = self.state[p]
				
				# Buffers:
				if 'sum_grad' not in param_state:
					param_state['sum_grad'] = torch.zeros_like(grads[-1]) # 1. momentum
					param_state['sum_grad_grad'] = torch.zeros_like(grads[-1]) # 2. momentum
				
				sum_grads.append(param_state['sum_grad']) # 1. momentum
				sum_grad_grads.append(param_state['sum_grad_grad']) # 2. momentum
				
			if len(thetas)==0:
				continue
		
			if weight_decay != 0:
				torch._foreach_add_(thetas,thetas,alpha=-lr*weight_decay)
			
			if clip_grad_adapt is not None and self.iteration > self.clip_grad_warm_up:
				thresholds = torch._foreach_clamp_min(
								torch._foreach_mul(
									torch._foreach_sqrt(
										torch._foreach_mul(sum_grad_grads,bias_correction2))
								,clip_grad_adapt)
							,clip_grad_min)
				torch._foreach_clamp_min_(grads,torch._foreach_neg(thresholds))
				torch._foreach_clamp_max_(grads,thresholds)
				del thresholds
			
			# 1. momentum
			torch._foreach_mul_(sum_grads,beta_1)
			torch._foreach_add_(sum_grads,grads,alpha=(1-beta_1))
			
			# 2. momentum
			torch._foreach_mul_(sum_grad_grads,beta_2)
			torch._foreach_addcmul_(sum_grad_grads,grads,grads,value=(1-beta_2))
			
			# apply update step
			torch._foreach_addcdiv_(thetas,sum_grads,
				torch._foreach_add(
					torch._foreach_sqrt(
						torch._foreach_mul(sum_grad_grads,bias_correction2))
				,self.eps)
			,value=lr_bias_correction1)

		return loss
