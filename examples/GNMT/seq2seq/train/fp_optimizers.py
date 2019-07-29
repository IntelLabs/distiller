import logging
import math

import torch
from torch.nn.utils import clip_grad_norm_


class Fp16Optimizer:

    @staticmethod
    def set_grads(params, params_with_grad):
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = torch.nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    @staticmethod
    def set_weights(params, new_params):
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)

    def __init__(self, fp16_model, grad_clip=float('inf'), loss_scale=8192,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=2048):
        logging.info('Initializing fp16 optimizer')
        self.initialize_model(fp16_model)

        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        self.fp16_model.zero_grad()
        self.fp32_params = [param.to(torch.float32).detach()
                            for param in model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer, update=True):
        loss *= self.loss_scale

        self.fp16_model.zero_grad()
        loss.backward()

        self.set_grads(self.fp32_params, self.fp16_model.parameters())
        if self.loss_scale != 1.0:
            for param in self.fp32_params:
                param.grad.data /= self.loss_scale

        norm = clip_grad_norm_(self.fp32_params, self.grad_clip)

        if update:
            if math.isfinite(norm):
                optimizer.step()
                self.set_weights(self.fp16_model.parameters(), self.fp32_params)
                self.since_last_invalid += 1
            else:
                self.loss_scale /= self.dls_downscale
                self.since_last_invalid = 0
                logging.info('Gradient norm: {}'.format(norm))
                logging.info('Skipped batch, new scale: {}'.format(self.loss_scale))

            if self.since_last_invalid >= self.dls_upscale_interval:
                self.loss_scale *= self.dls_upscale
                self.loss_scale = min(self.loss_scale, 8192.0)
                logging.info('Upscaling, new scale: {}'.format(self.loss_scale))
                self.since_last_invalid = 0


class Fp32Optimizer:

    def __init__(self, model, grad_clip=None):
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, update=True):
        loss.backward()
        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if update:
            optimizer.step()
        self.model.zero_grad()
