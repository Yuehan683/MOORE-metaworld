"""
Thin wrapper that runs the original MetaWorld SAC-MT script
but swaps in the SVD network module.

This keeps the original file untouched.
"""

import runpy
import sys

import moore.utils.networks_sac_svd as networks_sac_svd


def main():
    # Replace the module that the original script imports:
    #   import moore.utils.networks_sac as Network
    sys.modules["moore.utils.networks_sac"] = networks_sac_svd

    # Execute the original training script as __main__
    runpy.run_module("run_metaworld_sac_mt", run_name="__main__")


if __name__ == "__main__":
    main()