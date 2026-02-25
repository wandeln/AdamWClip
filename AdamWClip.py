import torch
from torch.optim.optimizer import Optimizer

class AdamWClip(Optimizer):
	"""
	AdamW optimizer with adaptive gradient clipping
	"""

	def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, clip_grad_adapt = 3, clip_grad_min=0.01, clip_grad_warm_up=10):
		"""
		params, lr, betas, eps, weight_decay is identical to AdamW optimizer
		On top of that, we have the following parameters:
		:clip_grad_adapt:   adaptive gradient clipping threshold in terms of standard deviations of the clipped gradient distribution (default: 3)
		:clip_grad_min:     minimum value for the adaptive gradient clipping threshold (default: 0.01)
		:clip_grad_warm_up: Number of initial update steps without gradient clipping to obtain reasonable gradient statistics at the beginning (default: 10)
		"""
		beta_1, beta_2 = betas
		defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2,weight_decay=weight_decay, clip_grad_adapt = clip_grad_adapt,clip_grad_min=clip_grad_min)
		self.iteration = 0
		self.clip_grad_warm_up = clip_grad_warm_up
		self.eps = eps
		super(AdamWClip, self).__init__(params, defaults)

	def step(self, closure=None, end=False):
		"""Performs a single optimization step."""
		loss = None
		if closure is not None:
			loss = closure()

		self.iteration += 1

		for group in self.param_groups:
			lr = group['lr']
			beta_1 = group['beta_1']
			beta_2 = group['beta_2']
			weight_decay = group['weight_decay']
			clip_grad_adapt = group['clip_grad_adapt']
			clip_grad_min = group['clip_grad_min']

			for p in group['params']:
				if p.grad is None:
					continue
				theta = p.data
				grad = p.grad.data
				if weight_decay != 0:
					theta.add_(-weight_decay*lr*theta)

				# Momentum part
				param_state = self.state[p]

				# Buffers:
				if 'sum_grad' not in param_state:
					# 1. momentum
					sum_grad = param_state['sum_grad'] = torch.zeros_like(grad)
					# 2. momentum
					sum_grad_grad = param_state['sum_grad_grad'] = torch.zeros_like(grad)
				else:
					# 1. momentum
					sum_grad = param_state['sum_grad']
					# 2. momentum
					sum_grad_grad = param_state['sum_grad_grad']

				if clip_grad_adapt is not None and self.iteration > self.clip_grad_warm_up:
					E_grad_grad = sum_grad_grad / (1-beta_2**self.iteration)
					clamp_adapt = clip_grad_adapt*torch.sqrt(E_grad_grad).clamp_(min=clip_grad_min)
					grad.clamp_(-clamp_adapt,clamp_adapt)

				# 1. momentum
				sum_grad.mul_(beta_1).add_(grad*(1-beta_1))

				# 2. momentum
				sum_grad_grad.mul_(beta_2).add_(grad*grad*(1-beta_2))

				# bias correction
				E_grad = sum_grad / (1-beta_1**self.iteration)
				E_grad_grad = sum_grad_grad / (1-beta_2**self.iteration) # could be optimized

				step = lr*E_grad/(torch.sqrt(E_grad_grad)+self.eps) # could be optimized

				# apply update step
				theta.add_(-step)

		return loss
