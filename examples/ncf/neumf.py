import numpy as np
import torch
import torch.nn as nn

import distiller.modules

import logging
import os
msglogger = logging.getLogger()


class NeuMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg,
                 mlp_layer_sizes, mlp_layer_regs, split_final=False):
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()

        self.mf_dim = mf_dim
        self.mlp_layer_sizes = mlp_layer_sizes

        nb_mlp_layers = len(mlp_layer_sizes)

        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)

        self.mf_mult = distiller.modules.EltwiseMult()
        self.mlp_concat = distiller.modules.Concat(dim=1)

        self.mlp = nn.ModuleList()
        self.mlp_relu = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501
            self.mlp_relu.extend([nn.ReLU()])

        self.split_final = split_final
        if not split_final:
            self.final_concat = distiller.modules.Concat(dim=1)
            self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)
        else:
            self.final_mlp = nn.Linear(mlp_layer_sizes[-1], 1)
            self.final_mf = nn.Linear(mf_dim, 1)
            self.final_add = distiller.modules.EltwiseAdd()

        self.sigmoid = nn.Sigmoid()

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        if not split_final:
            lecunn_uniform(self.final)
        else:
            lecunn_uniform(self.final_mlp)
            lecunn_uniform(self.final_mf)

    def load_state_dict(self, state_dict, strict=True):
        if 'final.weight' in state_dict and self.split_final:
            # Loading no-split checkpoint into split model

            # MF weights come first, then MLP
            final_weight = state_dict.pop('final.weight')
            state_dict['final_mf.weight'] = final_weight[0][:self.mf_dim].unsqueeze(0)
            state_dict['final_mlp.weight'] = final_weight[0][self.mf_dim:].unsqueeze(0)

            # Split bias 50-50
            final_bias = state_dict.pop('final.bias')
            state_dict['final_mf.bias'] = final_bias * 0.5
            state_dict['final_mlp.bias'] = final_bias * 0.5
        elif 'final_mf.weight' in state_dict and not self.split_final:
            # Loading split checkpoint into no-split model
            state_dict['final.weight'] = torch.cat((state_dict.pop('final_mf.weight')[0],
                                                    state_dict.pop('final_mlp.weight')[0])).unsqueeze(0)
            state_dict['final.bias'] = state_dict.pop('final_mf.bias') + state_dict.pop('final_mlp.bias')

        super(NeuMF, self).load_state_dict(state_dict, strict)

    def forward(self, user, item, sigmoid):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = self.mf_mult(xmfu, xmfi)

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = self.mlp_concat(xmlpu, xmlpi)
        for i, (layer, act) in enumerate(zip(self.mlp, self.mlp_relu)):
            xmlp = layer(xmlp)
            xmlp = act(xmlp)

        if not self.split_final:
            x = self.final_concat(xmf, xmlp)
            x = self.final(x)
        else:
            xmf = self.final_mf(xmf)
            xmlp = self.final_mlp(xmlp)
            x = self.final_add(xmf, xmlp)
        if sigmoid:
            x = self.sigmoid(x)
        return x
