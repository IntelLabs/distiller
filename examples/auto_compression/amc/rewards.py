#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import math


msglogger = logging.getLogger("examples.auto_compression.amc")


def reward_factory(reward_name):
    """Returns a reward function and a function with logic to clamp an action.

    This pair is defines the --amc-protocol choice.
    """
    return {
        "mac-constrained": (amc_mac_constrained_reward_fn, mac_constrained_clamp_action),
        "accuracy-guaranteed": (amc_accuracy_guarantee_reward_fn, None),
        "mac-constrained-experimental": (mac_constrained_experimental_reward_fn, None),
        "harmonic-mean": (harmonic_mean_reward_fn, None),
        "punish-agent": (punish_agent_reward_fn, None)
    }[reward_name]


def amc_mac_constrained_reward_fn(env, top1, top5, vloss, total_macs):
    return top1/100


def amc_accuracy_guarantee_reward_fn(env, top1, top5, vloss, total_macs):
    return -(1-top1/100) * math.log(total_macs)


def mac_constrained_experimental_reward_fn(env, top1, top5, vloss, total_macs):
    """A more intuitive reward for constraining the compute and optimizing the
    accuracy under this constraint.
    """
    macs_normalized = total_macs/env.original_model_macs
    reward = top1/100
    if macs_normalized > (env.amc_cfg.target_density+0.002):
        reward = -3 - macs_normalized
    else:
        reward += 1
    return reward


def mac_constrained_clamp_action(env, pruning_action):
    """Compute a resource-constrained action"""
    layer_macs = env.net_wrapper.layer_macs(env.current_layer())
    assert layer_macs > 0
    reduced = env.removed_macs
    prunable_rest, non_prunable_rest = env.rest_macs_raw()
    rest = prunable_rest * min(0.9, env.action_high)
    target_reduction = (1. - env.amc_cfg.target_density) * env.original_model_macs
    assert reduced == env.original_model_macs - env.net_wrapper.total_macs
    duty = target_reduction - (reduced + rest)
    pruning_action_final = min(1., max(pruning_action, duty/layer_macs))

    msglogger.debug("\t\tflops=%.3f  reduced=%.3f  rest=%.3f  duty=%.3f" % (layer_macs, reduced, rest, duty))
    msglogger.debug("\t\tpruning_action=%.3f  pruning_action_final=%.3f" % (pruning_action, pruning_action_final))
    msglogger.debug("\t\ttarget={:.2f} reduced={:.2f} rest={:.2f} duty={:.2f} flops={:.2f}\n".
                    format(1-env.amc_cfg.target_density, reduced/env.original_model_macs,
                           rest/env.original_model_macs,
                           duty/env.original_model_macs,
                           layer_macs/env.original_model_macs))
    if pruning_action_final != pruning_action:
        msglogger.debug("pruning_action={:.2f}==>pruning_action_final={:.2f}".format(pruning_action,
                                                                                     pruning_action_final))
    return pruning_action_final


def harmonic_mean_reward_fn(env, top1, top5, vloss, total_macs):
    """This reward is based on the idea of weighted harmonic mean

    Balance compute and accuracy provided a beta value that weighs the two components.
    See: https://en.wikipedia.org/wiki/F1_score
    """
    beta = 1
    #beta = 0.75  # How much to favor accuracy
    macs_normalized = total_macs/env.original_model_macs
    reward = (1 + beta**2) * top1/100 * macs_normalized / (beta**2 * macs_normalized + top1/100)
    return reward


def punish_agent_reward_fn(env, top1, top5, vloss, total_macs):
    """This reward punishes the agent when it produces networks that don't comply with the MACs resource-constraint,
    (the negative reward is in proportion to the network density).  Otherwise, the reward is the Top1 accuracy.
    """
    if not env.is_macs_constraint_achieved(total_macs):
        current_density = total_macs / env.original_model_macs
        reward = env.amc_cfg.target_density - current_density
    else:
        reward = top1/100
    return reward
