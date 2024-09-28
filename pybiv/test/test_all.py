from .test_approximate import (test_product, test_assembly,
                               test_array_vs_function, test_convergence)
from .test_tools import (test_rename_orient, test_get_c, test_get_nbrs)
from .test_trws import (test_trws_approx_inplace, test_trws_tree,
                        test_trws_consistency, test_trws_accuracy)
import numpy as np


def test_all():
    # test approximate
    test_product()
    test_assembly()
    test_array_vs_function()
    test_convergence()

    # test tools
    test_rename_orient()
    test_get_c()
    test_get_nbrs()

    # test trws
    test_trws_approx_inplace()
    test_trws_tree()
    test_trws_consistency()
    test_trws_accuracy()
