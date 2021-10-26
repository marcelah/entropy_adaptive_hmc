###########
#Class for a upper tridiagonal operator with efficient (linear)
#log-determinant calculation that is not implemented
#in the tridiagonal operator in the parent class in tensorflow
#Can be used as the Cholesky factor of a tridiagonal matrix
###########
import tensorflow as tf
from tensorflow.linalg import LinearOperatorTridiag
from tensorflow.python.ops import math_ops
_SEQUENCE = 'sequence'


class LinearOperatorUpperTridiag(LinearOperatorTridiag):

  def __init__(self,
               diagonal,
               super_diagonal,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name='LinearLowerOperatorTridiag'):

    diagonals = [super_diagonal, diagonal, tf.zeros_like(diagonal)]
    super(LinearOperatorUpperTridiag, self).__init__(
      diagonals = diagonals,
      diagonals_format = _SEQUENCE,
      is_non_singular = is_non_singular,
      is_self_adjoint = is_self_adjoint,
      is_positive_definite = is_positive_definite,
      is_square = is_square,
      name = name)

  def _log_abs_determinant(self):
    return math_ops.reduce_sum(math_ops.log(math_ops.abs(self._diag_part())), axis=[-1])