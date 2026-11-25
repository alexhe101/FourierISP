"""Optional BasicSR adapter. Install the standalone package with pip install -e ./modern."""
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from fourierisp import FourierISP
except ImportError:
    # BasicSR scans all architecture files, including for legacy experiments.
    # Do not break the original registry when the optional package is absent.
    FourierISP = None

if FourierISP is not None:
    @ARCH_REGISTRY.register()
    class FourierISPModern(FourierISP):
        def __init__(self, in_chn=3, wf=24, depth=4, relu_slope=0.2):
            if in_chn != 3:
                raise ValueError('FourierISP expects 3-channel demosaiced RGB')
            super().__init__(width=wf, depth=depth, relu_slope=relu_slope)

        def forward(self, x, demo):
            # Preserve the old list ABI for BasicSR model/loss wrappers.
            return list(super().forward(x, demo))
