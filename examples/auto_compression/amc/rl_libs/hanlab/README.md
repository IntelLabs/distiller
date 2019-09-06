The code in this directory originates from HAN Lab's [AMC github repository](https://github.com/mit-han-lab/amc-release).

We copied the DDPG files from [HAN Lab's github](https://github.com/mit-han-lab/amc-release) to `distiller/examples/auto_compression/amc/rl_libs/private/`.<br>
Specifically:
- `mit-han-lab/amc-release/tree/master/lib/agent.py` ==> `distiller/examples/auto_compression/amc/rl_libs/private`
- `mit-han-lab/amc-release/tree/master/lib/memory.py` ==> `distiller/examples/auto_compression/amc/rl_libs/private`
- `mit-han-lab/amc-release/tree/master/lib/utils.py` ==> `distiller/examples/auto_compression/amc/rl_libs/private`

Function `train()` was copied from `mit-han-lab/amc-release/tree/master/lib/amc_search.py` to the new file in `distiller/examples/auto_compression/amc/rl_libs/private/agent.py`.

Some non-functional changes were introduced in order for this to compile under Distiller.

The MIT license was copies to distiller/licenses/hanlab-amc-license.txt.

 