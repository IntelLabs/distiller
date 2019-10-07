def add_automl_args(argparser):
    """
    Helper function which defines command-line arguments specific to AMC.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    group = argparser.add_argument_group('AutoML Compression Arguments')
    group.add_argument('--amc-cfg', dest='amc_cfg_file', type=str, action='store',
                    help='AMC configuration file')
    group.add_argument('--amc-protocol', choices=["mac-constrained",
                                                  #"param-constrained",
                                                  "accuracy-guaranteed",
                                                  "mac-constrained-experimental",
                                                  "punish-agent"],
                       default="mac-constrained", help='Compression-policy search protocol')
    group.add_argument('--amc-ft-epochs', type=int, default=1,
                       help='The number of epochs to fine-tune each discovered network')
    group.add_argument('--amc-save-chkpts', action='store_true', default=False,
                       help='Save checkpoints of all discovered networks')
    group.add_argument('--amc-action-range',  type=float, nargs=2, default=[0.0, 0.80],
                       help='Density action range (a_min, a_max)')
    group.add_argument('--amc-heatup-episodes', type=int, default=100,
                       help='The number of episodes for heatup/exploration')
    group.add_argument('--amc-training-episodes', type=int, default=700,
                       help='The number of episodes for training/exploitation')
    group.add_argument('--amc-reward-frequency', type=int, default=None,
                       help='Reward computation frequency (measured in agent steps)')
    group.add_argument('--amc-target-density', type=float,
                       help='Target density of the network we are seeking')
    group.add_argument('--amc-agent-algo', choices=["ClippedPPO-continuous",
                                                    "ClippedPPO-discrete",
                                                    "TD3",
                                                    "DDPG",
                                                    "Random-policy"],
                       default="ClippedPPO-continuous",
                       help="The agent algorithm to use")
    group.add_argument('--amc-ft-frequency', type=int, default=None,
                       help='How many action-steps between fine-tuning.\n'
                       'By default there is no fine-tuning between steps.')
    group.add_argument('--amc-prune-pattern', choices=["filters", "channels"],
                       default="filters", help="The pruning pattern")
    group.add_argument('--amc-prune-method', choices=["l1-rank",
                                                      "stochastic-l1-rank",
                                                      "fm-reconstruction"],
                       default="l1-rank", help="The pruning method")
    group.add_argument('--amc-rllib', choices=["coach",
                                               "spinningup",
                                               "hanlab",
                                               "random"],
                       default=None, help="Choose which RL library to use")
    group.add_argument('--amc-group-size', type=int, default=1,
                       help="Number of filters/channels to group")
    group.add_argument('--amc-reconstruct-pts', dest="amc_fm_reconstruction_n_pts", type=int, default=10,
                       help="Number of filters/channels to group")
    group.add_argument('--amc-ranking-noise', type=float, default=0.,
                       help='Strcuture ranking noise')
    return argparser
