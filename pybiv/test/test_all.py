from .test_approximate import (test_product, test_assembly,
                               test_array_vs_function, test_convergence)
from .test_tools import (test_rename_orient, test_get_c, test_get_nbrs)
from .test_trws import (test_trws_approx_inplace, test_trws_tree,
                        test_trws_consistency, test_trws_accuracy)
from .test_cd import (test_cd_approx_inplace, test_cd_tree,
                        test_cd_consistency, test_cd_accuracy)
from .test_lpdlp import (test_lpdlp_approx_inplace, test_lpdlp_tree,
                        test_lpdlp_consistency, test_lpdlp_accuracy)
from .test_bcadtr import (test_bcadtr_approx_inplace, test_bcadtr_tree,
                        test_bcadtr_consistency, test_bcadtr_accuracy)
from .test_bcadetr import (test_bcadetr_approx_inplace, test_bcadetr_tree,
                        test_bcadetr_consistency, test_bcadetr_accuracy)


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

    # test cd
    test_cd_approx_inplace()
    test_cd_tree()
    test_cd_consistency()
    test_cd_accuracy()

    # test lpdlp
    test_lpdlp_approx_inplace()
    test_lpdlp_tree()
    test_lpdlp_consistency()
    test_lpdlp_accuracy()

    # test bcadtr
    test_bcadtr_approx_inplace()
    test_bcadtr_tree()
    test_bcadtr_consistency()
    test_bcadtr_accuracy()

    # test bcadetr
    test_bcadetr_approx_inplace()
    test_bcadetr_tree()
    test_bcadetr_consistency()
    test_bcadetr_accuracy()
