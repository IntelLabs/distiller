def add_automl_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command-line arguments for AMC to any application.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    argparser.add_argument('--amc', dest='AMC', action='store_true', help='AutoML Compression')
    group = argparser.add_argument_group('AutoML Compression Arguments')
    group.add_argument('--amc-cfg', dest='amc_cfg_file', type=str, action='store',
                    help='AMC configuration file')
    group.add_argument('--amc-protocol', choices=["mac-constrained",
                                                  "param-constrained",
                                                  "accuracy-guaranteed",
                                                  "mac-constrained-experimental"],
                       default="mac-constrained", help='Compression-policy search protocol')
    group.add_argument('--amc-ft-epochs', type=int, default=1,
                       help='The number of epochs to fine-tune each discovered network')
    group.add_argument('--amc-save-chkpts', action='store_true', default=False,
                       help='Save checkpoints of all discovered networks')
    group.add_argument('--amc-action-range',  type=float, nargs=2, default=[0.0, 0.80],
                       help='Density action range (a_min, a_max)')
    group.add_argument('--amc-heatup-epochs', type=int, default=100,
                       help='The number of epochs for heatup/exploration')
    group.add_argument('--amc-training-epochs', type=int, default=300,
                       help='The number of epochs for training/exploitation')
    group.add_argument('--amc-reward-frequency', type=int, default=None,
                       help='Reward computation frequency (measured in agent steps)')
    group.add_argument('--amc-target-density', type=float,
                       help='Target density of the network we are seeking')
    group.add_argument('--amc-agent-algo', choices=["ClippedPPO-continuous",
                                                    "ClippedPPO-discrete",
                                                    "DDPG",
                                                    "Random-policy"],
                       default="ClippedPPO-continuous",
                       help="The agent algorithm to use")
    # group.add_argument('--amc-thinning', action='store_true', default=False,
    #                    help='Perform netowrk thinning after altering each layer')
    group.add_argument('--amc-ft-frequency', type=int, default=None,
                       help='How many action-steps between fine-tuning.\n'
                       'By default there is no fine-tuning between steps.')
    group.add_argument('--amc-prune-pattern', choices=["filters", "channels"],
                       default="filters", help="The pruning pattern")
    group.add_argument('--amc-prune-method', choices=["l1-rank", "stochastic-l1-rank", "fm-reconstruction"],
                       default="l1-rank", help="The pruning method")
