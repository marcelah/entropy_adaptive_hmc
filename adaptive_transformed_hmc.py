"""Adaptiv Hamiltonian Monte Carlo algorithm with non-linear transformation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import metropolis_hastings
import leapfrog_integrator_trajectory as leapfrog_impl
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.internal import prefer_static
import pdb
import collections
from tensorflow_probability.python.random import rademacher
from tensorflow_probability.python.distributions import Geometric, Normal, MultivariateNormalDiag


__all__ = [
  'AdaptivePreconditionedHamiltonianMonteCarlo',
]



###
# code from tfp from transformed kernel
###

def make_log_det_jacobian_fn(bijector, direction):
  """Makes a function which applies a list of Bijectors' `log_det_jacobian`s."""
  if not mcmc_util.is_list_like(bijector):
    bijector = [bijector]
  attr = '{}_log_det_jacobian'.format(direction)
  def fn(state_parts, event_ndims):
    return [
      getattr(b, attr)(sp, event_ndims =e)
      for b, e, sp in zip(bijector, event_ndims, state_parts)
    ]
  return fn


def make_transform_fn(bijector, direction):
  """Makes a function which applies a list of Bijectors' `forward`s."""
  if not mcmc_util.is_list_like(bijector):
    bijector = [bijector]
  def fn(state_parts):
    return [getattr(b, direction)(sp) for b, sp in zip(bijector, state_parts)]
  return fn


def make_transformed_log_prob(
  log_prob_fn, bijector, direction, enable_bijector_caching =True):
  """Transforms a log_prob function using bijectors.

  Note: `direction = 'inverse'` corresponds to the transformation calculation
  done in `tfp.distributions.TransformedDistribution.log_prob`.

  Args:
    log_prob_fn: Python `callable` taking an argument for each state part which
      returns a `Tensor` representing the joint `log` probability of those state
      parts.
    bijector: `tfp.bijectors.Bijector`-like instance (or list thereof)
      corresponding to each state part. When `direction = 'forward'` the
      `Bijector`-like instance must possess members `forward` and
      `forward_log_det_jacobian` (and corresponding when
      `direction = 'inverse'`).
    direction: Python `str` being either `'forward'` or `'inverse'` which
      indicates the nature of the bijector transformation applied to each state
      part.
    enable_bijector_caching: Python `bool` indicating if `Bijector` caching
      should be invalidated.
      Default value: `True`.

  Returns:
    transformed_log_prob_fn: Python `callable` which takes an argument for each
      transformed state part and returns a `Tensor` representing the joint `log`
      probability of the transformed state parts.
  """
  if direction not in {'forward', 'inverse'}:
    raise ValueError('Argument `direction` must be either `"forward"` or '
                     '`"inverse"`; saw "{}".'.format(direction))
  fn = make_transform_fn(bijector, direction)
  ldj_fn = make_log_det_jacobian_fn(bijector, direction)
  def transformed_log_prob_fn(*state_parts):
    """Log prob of the transformed state."""
    if not enable_bijector_caching:
      state_parts = [tf.identity(sp) for sp in state_parts]
    tlp = log_prob_fn(*fn(state_parts))
    tlp_rank = prefer_static.rank(tlp)
    event_ndims = [(prefer_static.rank(sp) - tlp_rank) for sp in state_parts]
    return tlp + sum(ldj_fn(state_parts, event_ndims))
  return transformed_log_prob_fn






class AdaptiveKernelResults(
  mcmc_util.PrettyNamedTupleMixin,
  collections.namedtuple('AdaptiveKernelResults',
                         [
                           'transformed_state',
                           'log_acceptance_correction',
                           'target_log_prob',
                           'eigenvalue',
                           'proposal_log_prob'
                         ])):
  __slots__ = ()

class AdaptivePreconditionedHamiltonianMonteCarlo(hmc.HamiltonianMonteCarlo):
  """Runs Hamiltonian Monte Carlo with a non-identity mass matrix."""

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               bijector_fn,
               seed =None,
               name =None,
               learning_rate = .005,
               opt_acceptance_rate = .65,
               biased_accept_grads = True,
               num_exact_trace_terms = 2,
               num_trace_terms_probs = .5,
               beta_learning_rate = .01,
               beta_min = .001,
               beta_max = 100,
               clip_grad_value = 1000.,
               penalty_fn = None
               ):
    """Initializes this transition kernel.
    """
    self._seed_stream = SeedStream(seed, salt ='hmc')
    uhmc_kwargs = {} if seed is None else dict(seed =self._seed_stream())
    mh_kwargs = {} if seed is None else dict(seed =self._seed_stream())
    self._impl = metropolis_hastings.MetropolisHastings(
      inner_kernel =UncalibratedAdaptivePreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn =target_log_prob_fn,
        step_size =step_size,
        num_leapfrog_steps =num_leapfrog_steps,
        bijector_fn =bijector_fn,
        name =name or 'hmc_kernel',
        learning_rate = learning_rate,
        opt_acceptance_rate = opt_acceptance_rate,
        num_exact_trace_terms = num_exact_trace_terms,
        num_trace_terms_probs = num_trace_terms_probs,
        beta_learning_rate = beta_learning_rate,
        beta_min = beta_min,
        beta_max = beta_max,
        clip_grad_value = clip_grad_value,
        biased_accept_grads = biased_accept_grads,
        penalty_fn = penalty_fn,
        **uhmc_kwargs),
      **mh_kwargs)
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters['step_size_update_fn'] = None
    self._parameters['seed'] = seed

  def one_step(self, current_state, previous_kernel_results, seed = None):

    # update transformed state so that it is the inverse of the current
    # using the bijector with updated params
    previous_transformed_state_new_param = self._impl.inner_kernel.bijector_fn.inverse(current_state)
    # update accepted results log prob based on new params
    transformed_target_log_prob_fn = make_transformed_log_prob(
      self._impl.inner_kernel.target_log_prob_fn,
      self._impl.inner_kernel.bijector_fn,
      direction = 'forward',
      # TODO(b/72831017): Disable caching until gradient linkage
      # generally works.
      enable_bijector_caching = False)
    updated_log_prob = transformed_target_log_prob_fn(
      previous_transformed_state_new_param)

    previous_kernel_results = previous_kernel_results._replace(
      accepted_results = previous_kernel_results.accepted_results._replace(
        target_log_prob = transformed_target_log_prob_fn(
      previous_transformed_state_new_param),
        transformed_state = previous_transformed_state_new_param))

    #tf.debugging.assert_all_finite(current_state,'current_state')
    #tf.debugging.assert_all_finite(previous_transformed_state_new_param,'previous_transformed_state_new_param')
    #tf.debugging.assert_all_finite(updated_log_prob,'updated_log_prob')


    return super().one_step(current_state, previous_kernel_results, seed)


class UncalibratedAdaptivePreconditionedHamiltonianMonteCarlo(
  hmc.UncalibratedHamiltonianMonteCarlo):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `PreconditionedHamiltonianMonteCarlo(...)` or
  `MetropolisHastings(UncalibratedPreconditionedHamiltonianMonteCarlo(...))`.

  For more details on `UncalibratedPreconditionedHamiltonianMonteCarlo`, see
  `PreconditionedHamiltonianMonteCarlo`.
  """

  @mcmc_util.set_doc(hmc.UncalibratedHamiltonianMonteCarlo.__init__)
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               bijector_fn,
               learning_rate,
               beta_learning_rate,
               beta_min,
               beta_max,
               opt_acceptance_rate,
               num_exact_trace_terms,
               num_trace_terms_probs,
               biased_accept_grads,
               clip_grad_value,
               penalty_fn,
               name =None,
               ):
    super(UncalibratedAdaptivePreconditionedHamiltonianMonteCarlo, self).__init__(
      target_log_prob_fn,
      step_size,
      num_leapfrog_steps,
      # seed=None,
      name =name
    )
    self._parameters['bijector_fn'] = bijector_fn
    self._optimizer_pre_cond = tf.keras.optimizers.Adam(learning_rate =learning_rate)
    self._learning_rate = learning_rate
    self._num_exact_trace_terms = num_exact_trace_terms
    self._num_trace_terms_probs = num_trace_terms_probs
    self._opt_acceptance_rate = opt_acceptance_rate
    self._beta_learning_rate = beta_learning_rate
    self._step_size = tf.Variable(initial_value = step_size, trainable = False, name ='step_size')
    self._beta = tf.Variable(initial_value = 1., trainable = False, name ='beta')
    self._penalty_param = tf.Variable(initial_value = 1000., trainable = False, name ='penalty_param')
    self._beta_min = beta_min
    self._beta_max = beta_max
    self._clip_grad_value = clip_grad_value
    self._biased_accept_grads = biased_accept_grads

    if penalty_fn is None:
      def penalty_fn(evs):
        delta_1 = .75
        delta_2 = 1.75
        return tf.nest.map_structure(
          lambda ev: tf.where(abs(ev) < delta_2,
                              tf.where(abs(ev) < delta_1,
                                       tf.zeros_like(ev),
                                       (abs(ev) - delta_1) ** 2),
                              (delta_2 - delta_1) ** 2 + (delta_2 - delta_1) * (abs(ev) - delta_2)
                              ),
          evs
        )
    self._penalty_fn = penalty_fn


  @property
  def bijector_fn(self):
    return self._parameters['bijector_fn']


  @mcmc_util.set_doc(hmc.HamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed =None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'hmc', 'one_step')):

      #update previous results to adjust for new bijector params so that
      #current_state and previous_kernel_results.transformed_states
      #are linked through updated bijector
      transformed_current_state = self.bijector_fn.inverse(current_state)

      #transformed_current_state = tf.stop_gradient(previous_kernel_results.transformed_state)
      with tf.GradientTape(persistent = True) as tape1:
        bijector = self.bijector_fn
        step_size = self._step_size
        num_leapfrog_steps = self.num_leapfrog_steps


        transformed_target_log_prob_fn = make_transformed_log_prob(
          self.target_log_prob_fn,
          bijector,
          direction = 'forward',
          # TODO(b/72831017): Disable caching until gradient linkage
          # generally works.
          enable_bijector_caching = False)

        # momentum_distribution = MultivariateNormalDiag(loc=tf.zeros_like(current_state))

        [
          current_transformed_state_parts,
          step_sizes,
          momentum_distribution,
          current_transformed_target_log_prob,
          current_transformed_target_log_prob_grad_parts,
        ] = _prepare_args(
          transformed_target_log_prob_fn,
          transformed_current_state,
          step_size,
          maybe_expand =True)

        #tf.print('current_transformed_state_parts' ,current_transformed_state_parts)
        #tf.print('current_state' ,current_state)
        #tf.print('current_transformed_target_log_prob' ,current_transformed_target_log_prob)

        seed = samplers.sanitize_seed(seed)
        # TO DO: update seeds with new tfp version
        seeds = samplers.split_seed(seed, n = len(current_transformed_state_parts))

        current_momentum_noise_parts = tf.nest.map_structure(
          lambda x ,s: tf.cast(samplers.normal(shape =tf.shape(x), seed =s), x.dtype),
          current_transformed_state_parts, seeds)
        tape1.watch(current_momentum_noise_parts)

        current_momentum_parts = current_momentum_noise_parts

        momentum_log_prob = getattr(momentum_distribution,
                                    '_log_prob_unnormalized',
                                    momentum_distribution.log_prob)
        kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)


        leapfrog_kinetic_energy_fn = kinetic_energy_fn

        integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
          transformed_target_log_prob_fn, step_sizes, num_leapfrog_steps)

        [
          next_momentum_parts,
          next_transformed_state_parts,
          next_transformed_target_log_prob,
          _,
          _,
          _,
          transformed_state_parts_array
        ] = integrator(
          current_momentum_parts,
          current_transformed_state_parts,
          target =current_transformed_target_log_prob,
          target_grad_parts =current_transformed_target_log_prob_grad_parts,
          kinetic_energy_fn =leapfrog_kinetic_energy_fn,
        )

        transformed_state_parts_array = tf.nest.map_structure(
          lambda x: tf.ensure_shape(x, [num_leapfrog_steps+1]+x.shape[1:]), transformed_state_parts_array)

        def maybe_flatten(x):
          return x if mcmc_util.is_list_like(current_state) else x[0]

        dims = tf.nest.map_structure(
          lambda x: tf.cast(tf.shape(x)[1:], x.dtype), current_transformed_state_parts)
        tlp_rank = prefer_static.rank(current_transformed_target_log_prob)
        event_ndims = [(prefer_static.rank(sp) - tlp_rank) for sp in current_transformed_state_parts]

        ###
        # compute objective for optimizing log accept ratio
        ###
        if True:
          # if not self._biased_accept_grads:
          log_acceptance_correction = _compute_log_acceptance_correction(
            kinetic_energy_fn, current_momentum_parts,
            next_momentum_parts)
          #tf.print('log_acceptance_correction' ,log_acceptance_correction)
          # compute log-acceptance rates
          to_sum = [next_transformed_target_log_prob,
                    -current_transformed_target_log_prob,
                    log_acceptance_correction]
          log_accept_ratio = mcmc_util.safe_sum(
            to_sum, name = 'log_accept_ratio')
          log_accept_ratio_loss = -tf.minimum(tf.zeros([], dtype = log_accept_ratio.dtype), log_accept_ratio)
          #tf.print('log_accept_ratio', log_accept_ratio)

        # alternative stopped terms using Xi function
        if True:
          reparam_transformed_current_state = make_transform_fn(bijector, 'inverse')(
            current_state if mcmc_util.is_list_like(current_state) else [current_state])

          transformed_state_parts_array_stopped = tf.nest.map_structure(
            lambda q: tf.stop_gradient(q), transformed_state_parts_array)
          with tf.GradientTape() as tape2:
            tape2.watch(transformed_state_parts_array_stopped)
            _, target_log_prob_grad_parts_array = mcmc_util.maybe_call_fn_and_grads(
              self.target_log_prob_fn,
              make_transform_fn(bijector, 'forward')(transformed_state_parts_array_stopped),
              None, None)
            y = make_transform_fn(bijector, 'forward')(transformed_state_parts_array_stopped)
          transformed_target_log_prob_grad_parts_array= tape2.gradient(y, transformed_state_parts_array_stopped,
                                                                       tf.nest.map_structure
                                                                          (lambda q: tf.stop_gradient(q), target_log_prob_grad_parts_array))


          if not bijector.is_constant_jacobian:
            _, ldj_grads_array = mcmc_util.maybe_call_fn_and_grads(
              lambda *z: make_log_det_jacobian_fn(bijector, 'forward')(z, event_ndims),
              transformed_state_parts_array_stopped, None, None)
            transformed_target_log_prob_grad_parts_array = tf.nest.map_structure(
              lambda x ,y: x+ y, transformed_target_log_prob_grad_parts_array, ldj_grads_array)

          # stop gradients
          # transformed_target_log_prob_grad_parts_array = tf.nest.map_structure(
          #  lambda x: tf.stop_gradient(x), transformed_target_log_prob_grad_parts_array_non_stopped)

          sum_potential_grads = tf.nest.map_structure(
            lambda g: -.5 * (g[0] + g[-1]) - tf.reduce_sum(g[1:-1], 0), transformed_target_log_prob_grad_parts_array)
          kinetic_energy_error_stopped = mcmc_util.safe_sum(tf.nest.map_structure(
            lambda h, Us, v: (kinetic_energy_fn(v - h * Us) - kinetic_energy_fn(v)),
            step_sizes, sum_potential_grads, current_momentum_noise_parts))
          log_acceptance_correction_stopped = tf.nest.map_structure(lambda x: -x, kinetic_energy_error_stopped)

          weights = tf.cast(num_leapfrog_steps - tf.range(1, num_leapfrog_steps),
                            current_momentum_noise_parts[0].dtype)
          xi = tf.nest.map_structure(
            lambda g: tf.einsum('lsd,l->sd', -g[1:-1], weights), transformed_target_log_prob_grad_parts_array)
          reparam_transformed_next_state = tf.nest.map_structure(
            lambda x, v, h, g, gs: x + h * num_leapfrog_steps * v - h * h * g + h * h / 2 * num_leapfrog_steps * gs[0],
            reparam_transformed_current_state, current_momentum_noise_parts, step_sizes,
            xi, transformed_target_log_prob_grad_parts_array)
          reparam_transformed_next_target_log_prob, _ = mcmc_util.maybe_call_fn_and_grads(
            transformed_target_log_prob_fn, reparam_transformed_next_state, None, None)
          reparam_current_transformed_target_log_prob, _ = mcmc_util.maybe_call_fn_and_grads(
            transformed_target_log_prob_fn, reparam_transformed_current_state, None, None)
          potential_energy_error_stopped = -(reparam_transformed_next_target_log_prob -
                                             reparam_current_transformed_target_log_prob)
          log_accept_loss_stopped = - tf.minimum(tf.zeros([], potential_energy_error_stopped.dtype),
                                                 -potential_energy_error_stopped - kinetic_energy_error_stopped)
          #tf.print('xi',xi)
          #tf.print('reparam_transformed_next_state', reparam_transformed_next_state)
          #tf.print('reparam_transformed_next_target_log_prob',reparam_transformed_next_target_log_prob)
          #tf.print('reparam_current_transformed_target_log_prob',reparam_current_transformed_target_log_prob)
          #tf.print('log_accept_loss_stopped', log_accept_loss_stopped)
          #tf.print('kinetic_energy_error_stopped', kinetic_energy_error_stopped)
          #tf.print('potential_energy_error_stopped', potential_energy_error_stopped)
          log_accept_ratio_stopped = -potential_energy_error_stopped - kinetic_energy_error_stopped
          #tf.print('log_accept_ratio_stopped', log_accept_ratio_stopped)

        #############
        # Approximate DS matrix using a constant Hessian at the mid-point
        #############

        def approximate_DS_matrix_vector_first_order(ws):

          state_parts_mid = make_transform_fn(bijector, 'forward')(
            tf.nest.map_structure(lambda qs: qs[self.num_leapfrog_steps // 2], transformed_state_parts_array))
          state_parts_mid = tf.nest.map_structure(lambda q: tf.stop_gradient(q), state_parts_mid)
          reparam_transformed_state_parts_mid = make_transform_fn(bijector, 'inverse')(state_parts_mid)
          approx_DSw = tf.nest.map_structure(
            lambda h, q, w: h * h * tf.cast(self.num_leapfrog_steps ** 2 - 1, q.dtype) / 6. * (
              _target_hessian_vector_product(transformed_target_log_prob_fn,q,w)),
            step_sizes, reparam_transformed_state_parts_mid, ws)
          return approx_DSw



        if self.num_leapfrog_steps > 1:

          DSw_approximation = approximate_DS_matrix_vector_first_order
          ###########
          # Russian Roulette estimator for the residual part
          ###########
          # distribution of truncation level
          random_trace_terms_dist = Geometric(probs = self._num_trace_terms_probs)
          coeff_fn = lambda k: 1. / (1 - random_trace_terms_dist.cdf(
            k - self._num_exact_trace_terms + .1))
          trace_noise_parts = tf.nest.map_structure(
            lambda x, s: rademacher(shape = tf.shape(x), seed = s, dtype = x.dtype),
            current_transformed_state_parts, seeds)

          def loop_residual_trace_neumann(k, prev_jvp, ns_jvp, sum_jvp_entropy):
            new_jvp = DSw_approximation(prev_jvp)
            # clip values
            prev_jvp_norm = tf.nest.map_structure(lambda x: tf.linalg.norm(x, axis = -1, keepdims = True), prev_jvp)
            new_jvp_norm = tf.nest.map_structure(lambda x: tf.linalg.norm(x, axis = -1, keepdims = True), new_jvp)
            new_jvp = tf.nest.map_structure(
              lambda x, y, z: x * tf.minimum(tf.ones([], dtype = y.dtype), .99 * y / z),
              new_jvp, prev_jvp_norm, new_jvp_norm)

            new_ns_jvp = tf.nest.map_structure(
              lambda ns, jvp: ns + tf.cast(tf.pow(-1., k), ns.dtype) * tf.cast(coeff_fn(k), ns.dtype) * jvp, ns_jvp,
              new_jvp)
            new_sum_vjp_entropy = tf.nest.map_structure(
              lambda ent, jvp: ent + tf.cast(tf.pow(-1., k + 1.) / k * coeff_fn(k), jvp.dtype) * jvp,
              sum_jvp_entropy, new_jvp)

            return k + 1, new_jvp, new_ns_jvp, new_sum_vjp_entropy

          # random truncation level for russian roulette estimator
          sample_random_trace_terms = random_trace_terms_dist.sample(seed = seeds[0])
          loop_trace_noise_parts = trace_noise_parts
          with tape1.stop_recording():
            with tf.name_scope('russian_roulette_estimator'):
              _, jvp_residual, ns_jvp_residual, sum_jvp_residual_entropy = tf.while_loop(
                cond = lambda k, _0, _1, _2: k <= self._num_exact_trace_terms + sample_random_trace_terms,
                body = loop_residual_trace_neumann,
                loop_vars = [tf.constant(1., dtype = tf.float32),
                             loop_trace_noise_parts,
                             loop_trace_noise_parts,
                             [tf.zeros_like(ep) for ep in loop_trace_noise_parts]
                             ]
              )

          # power itertation for penalty lipschitz constant estimate
          # add some noise for stability, the noise might be dominant for very small jvps,
          # but then we should have a contraction, also clip value
          stable_jvp_residual = tf.nest.map_structure(lambda w: tf.clip_by_value(w + tf.cast(1e-6 * samplers.normal(
            w.shape), w.dtype), -1e10, 1e10), jvp_residual)
          # tf.print('stable_jvp_residual', stable_jvp_residual)
          normalised_power_iteration_vector = tf.nest.map_structure(
            lambda w: tf.stop_gradient(w / tf.linalg.norm(w, axis = -1, keepdims = True)), stable_jvp_residual)
          #tf.print('normalised_power_iteration_vector', normalised_power_iteration_vector)
          jvp_lipschitz = DSw_approximation(normalised_power_iteration_vector)


          max_eigenvalue_DS_lipschitz = tf.nest.map_structure(
            lambda w, DSw: tf.einsum('si,si->s', w, DSw),
            normalised_power_iteration_vector, jvp_lipschitz)
          #tf.print('max_eigenvalue_DS_lipschitz', max_eigenvalue_DS_lipschitz)

          # JVP using AD through tape
          jvp_noise = DSw_approximation(trace_noise_parts)
          neumann_trace_residual_for_grad = tf.nest.map_structure(
            lambda ns, j: tf.einsum('ij,ij->i', tf.stop_gradient(ns), j),
            ns_jvp_residual, jvp_noise)

          penalty = tf.nest.map_structure(lambda x: tf.cast(self._penalty_param, x.dtype) * x,
                                          self._penalty_fn(max_eigenvalue_DS_lipschitz))
          #tf.print('penalty', penalty)

          trace_residual_log_det = tf.nest.map_structure(
            lambda s, eps: tf.einsum('ij,ij->i', tf.stop_gradient(s), eps), sum_jvp_residual_entropy,
            trace_noise_parts)

          trace_estimate_for_grad = tf.nest.map_structure(lambda x, y: x - y,
                                                          neumann_trace_residual_for_grad, penalty)
          #tf.print('trace_estimate_for_grad', trace_estimate_for_grad)

        else:
          trace_estimate_for_grad = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0], c.dtype),
                                                          next_transformed_state_parts)
          trace_residual_log_det = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0], c.dtype),
                                                         next_transformed_state_parts)
          penalty = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0], c.dtype), next_transformed_state_parts)
          max_eigenvalue_DS_lipschitz = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0], c.dtype),
                                                              next_transformed_state_parts)


        ######
        # proposal entropy terms
        ######
        # linear/langevin part for proposal log prob
        #next_state_parts = make_transform_fn(bijector, 'forward')(next_transformed_state_parts)

        log_det_linear_term = tf.nest.map_structure(
          lambda d, h,: d * tf.math.log( h *self.num_leapfrog_steps),
          dims, step_sizes)
        #log_det_jacobian_transformation = make_log_det_jacobian_fn(bijector, 'forward')(
        #  next_transformed_state_parts, event_ndims)
        #next_state_parts = tf.nest.map_structure(lambda q: tf.stop_gradient(q), next_state_parts)
        log_det_jacobian_transformation = make_log_det_jacobian_fn(bijector, 'forward')(
          reparam_transformed_next_state, event_ndims)
        #tf.print('log_det_jacobian_transformation' ,log_det_jacobian_transformation)




        noise_proposal_log_probs = tf.nest.map_structure(
          lambda v: tf.reduce_sum(Normal(loc = 0. * tf.ones([], v.dtype), scale = 1.).log_prob(v), -1),
          current_momentum_noise_parts)

        proposal_log_prob_estimate = mcmc_util.safe_sum(tf.nest.map_structure(
          lambda y, j, z, c: - y - j - z + c,
          log_det_linear_term, log_det_jacobian_transformation, trace_residual_log_det, noise_proposal_log_probs))

        proposal_log_prob_for_grad = mcmc_util.safe_sum(tf.nest.map_structure(
          lambda y, j, z: - y - j - z,
          log_det_linear_term, log_det_jacobian_transformation, trace_estimate_for_grad))

        ######
        # loss for optimizing
        ######
        if self._biased_accept_grads:
          mcmc_loss = log_accept_loss_stopped + \
                      tf.cast(self._beta * tf.ones([]), log_accept_loss_stopped.dtype) * proposal_log_prob_for_grad
        else:
          mcmc_loss = log_accept_ratio_loss + \
                      tf.cast(self._beta * tf.ones([]), log_accept_ratio_loss.dtype) * proposal_log_prob_for_grad

          # mcmc_loss = -log_accept_ratio +  proposal_log_prob_for_grad
        #tf.print('proposal_log_prob_estimate', proposal_log_prob_estimate)


      #tf.print('mcmc_loss', mcmc_loss)
      #tf.debugging.assert_all_finite(mcmc_loss, 'mcmc_loss')

      next_state_parts = make_transform_fn(bijector, 'forward')(next_transformed_state_parts)

      # pdb.set_trace()
      grads = tape1.gradient(mcmc_loss, tape1.watched_variables())
      transformed_target_log_probs, _ = mcmc_util.maybe_call_fn_and_grads(
        transformed_target_log_prob_fn,
        transformed_state_parts_array, None, None)
      #tf.print('transformed_target_log_probs', transformed_target_log_probs)
      #tf.print('transformed_state_parts_array', transformed_state_parts_array)
      #tf.print('log_accept_grads', tape1.gradient(log_accept_ratio_loss, tape1.watched_variables()))
      #tf.print('stopped_log_accept_grads', tape1.gradient(log_accept_loss_stopped, tape1.watched_variables()))
      #tf.print('log det DF', tape1.gradient(log_det_jacobian_transformation, tape1.watched_variables()))

      # replace nans
      grads = tf.nest.map_structure(
        lambda g: tf.where(tf.math.is_nan(g), tf.zeros_like(g), g), grads)
      # clip grads
      grads = tf.nest.map_structure(lambda g: tf.clip_by_value(g, -self._clip_grad_value, self._clip_grad_value),
                                    grads)


      # decrease step_size if nan loss
      h_new = tf.cond(tf.math.is_finite(tf.reduce_sum(mcmc_loss)),
                      lambda: self._step_size,
                      lambda: self._step_size * (1 - self._learning_rate)
                      )
      self._step_size.assign(h_new)
      #tf.print('step size', self._step_size)


      # adapt beta
      if self._biased_accept_grads:
        acceptance_rate = tf.reduce_mean(tf.math.exp(tf.minimum(tf.zeros([], dtype = log_accept_ratio_stopped.dtype),
                                                                log_accept_ratio_stopped)))
      else:
        acceptance_rate = tf.reduce_mean(tf.math.exp(tf.minimum(tf.zeros([], dtype = log_accept_ratio.dtype),
                                                                log_accept_ratio)))
      beta = self._beta * tf.cast(
        1. + self._beta_learning_rate * tf.cast(
          acceptance_rate - self._opt_acceptance_rate, acceptance_rate.dtype),
        self._beta.dtype)
      beta = tf.where(tf.math.is_nan(beta), self._beta, beta)
      beta = tf.clip_by_value(beta, self._beta_min, self._beta_max)
      self._beta.assign(beta)
      #tf.print('acceptance_rate', acceptance_rate)
      #tf.print('beta', self._beta)

      # update penalty param
      current_penalty_param = self._penalty_param
      new_penalty_param = tf.clip_by_value(
        current_penalty_param + self._beta_learning_rate * tf.clip_by_value(
          tf.cast(tf.reduce_mean(penalty), tf.float32), 0, 1e2), 1., 1e4)
      new_penalty_param = tf.where(tf.math.is_nan(new_penalty_param),
                                   current_penalty_param + 100,
                                   new_penalty_param)
      self._penalty_param.assign(new_penalty_param)
      #tf.print('_penalty_param', self._penalty_param)

      #tf.print('params', tape1.watched_variables())
      #tf.print('grads', grads)

      # tf.print('penalty_grads', tape1.gradient(penalty, self.params))
    #pdb.set_trace()
    next_transformed_state = maybe_flatten(next_transformed_state_parts)
    new_kernel_results = previous_kernel_results._replace(
      log_acceptance_correction = log_acceptance_correction if not self._biased_accept_grads \
        else log_acceptance_correction_stopped,
      target_log_prob = next_transformed_target_log_prob,
      transformed_state = next_transformed_state,
      eigenvalue = mcmc_util.safe_sum(max_eigenvalue_DS_lipschitz),
      proposal_log_prob = proposal_log_prob_estimate
    )
    #tf.print('next_transformed_state_parts', next_transformed_state_parts)
    #tf.print('next_state_parts', next_state_parts)

    # update params
    self._optimizer_pre_cond.apply_gradients(zip(grads, tape1.watched_variables()))

    return maybe_flatten(next_state_parts), new_kernel_results


  def bootstrap_results(self, transformed_init_state):
    #with tf.GradientTape(persistent = False) as tape1:
    transformed_target_log_prob_fn = make_transformed_log_prob(
        self.target_log_prob_fn,
        self.bijector_fn,
        direction = 'forward',
        # TODO(b/72831017): Disable caching until gradient linkage
        # generally works.
        enable_bijector_caching = False)
    transformed_target_log_prob = transformed_target_log_prob_fn(transformed_init_state)
    hmc_results = super().bootstrap_results(transformed_init_state)
    if mcmc_util.is_list_like(transformed_init_state):
      transformed_init_state = [
        tf.convert_to_tensor(s, name = 'transformed_init_state')
        for s in transformed_init_state
      ]
    else:
      transformed_init_state = tf.convert_to_tensor(
        value = transformed_init_state, name = 'transformed_init_state')
    kernel_results = AdaptiveKernelResults(
      log_acceptance_correction = hmc_results.log_acceptance_correction,
      target_log_prob = transformed_target_log_prob,
      transformed_state = transformed_init_state,
      eigenvalue = transformed_target_log_prob,
      proposal_log_prob = transformed_target_log_prob
    )
    return kernel_results


def _compute_log_acceptance_correction(kinetic_energy_fn,
                                       current_momentums,
                                       proposed_momentums,
                                       name = None):
  """Helper to `kernel` which computes the log acceptance-correction.
  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:
  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```
  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.
  Inserting this into the detailed balance equation implies:
  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```
  One definition of `a(x'|x)` which satisfies (*) is:
  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```
  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)
  We call the bracketed term the "acceptance correction".
  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Given a probability density of `m(z)` for momentums, the chain eventually
  converges to:
  ```none
  p([x, z]) propto= target_prob(x) m(z)
  ```
  Relating this back to Metropolis-Hastings parlance, for HMC we have:
  ```none
  p([x, z]) propto= target_prob(x) m(z)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```
  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:
  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [m(z') / m(z)]
                       target_prob(x)
  ```
  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)
  For consistency, we compute this correction in log space, using the kinetic
  energy function, `K(z)`, which is the negative log probability of the momentum
  distribution. So the log acceptance probability is
  ```none
  log(correction) = log(m(z')) - log(m(z))
                  = K(z) - K(z')
  ```
  Note that this is equality, since the normalization constants on `m` cancel
  out.
  Args:
    kinetic_energy_fn: Python callable that can evaluate the kinetic energy
      of the given momentum. This is typically the negative log probability of
      the distribution over the momentum.
    current_momentums: (List of) `Tensor`s representing the value(s) of the
      current momentum(s) of the state (parts).
    proposed_momentums: (List of) `Tensor`s representing the value(s) of the
      proposed momentum(s) of the state (parts).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').
  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    current_kinetic = kinetic_energy_fn(current_momentums)
    proposed_kinetic = kinetic_energy_fn(proposed_momentums)
    return mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(transformed_target_log_prob_fn,
                  transformed_state,
                  step_size,
                  maybe_expand = False):
  """Helper which processes input args to meet list-like assumptions."""
  transformed_state_parts, _ = mcmc_util.prepare_state_parts(transformed_state, name = 'current_state')

  transformed_target_log_prob, grads_transformed_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
    transformed_target_log_prob_fn, transformed_state_parts, None, None)
  step_sizes, _ = mcmc_util.prepare_state_parts(
    step_size, dtype = transformed_target_log_prob.dtype, name = 'step_size')

  batch_rank = ps.rank(transformed_target_log_prob)

  def _batched_isotropic_normal_like(state_part):
    event_ndims = ps.rank(state_part) - batch_rank
    return independent.Independent(
      normal.Normal(ps.zeros_like(state_part, tf.float32), 1.),
      reinterpreted_batch_ndims = event_ndims)

  momentum_distribution = jds.JointDistributionSequential(
    [_batched_isotropic_normal_like(state_part)
     for state_part in transformed_state_parts])

  # The momentum will get "maybe listified" to zip with the state parts,
  # and this step makes sure that the momentum distribution will have the
  # same "maybe listified" underlying shape.
  if not mcmc_util.is_list_like(momentum_distribution.dtype):
    momentum_distribution = jds.JointDistributionSequential(
      [momentum_distribution])

  if len(step_sizes) == 1:
    step_sizes *= len(transformed_state_parts)
  if len(transformed_state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')

  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(transformed_state) else x[0]

  return [
    maybe_flatten(transformed_state_parts),
    maybe_flatten(step_sizes),
    momentum_distribution,
    transformed_target_log_prob,
    grads_transformed_target_log_prob,
  ]




def _target_hessian_vector_product(transformed_target_log_prob_fn, q, w):
  with tf.GradientTape() as tape1:
    tape1.watch(q)
    with tf.GradientTape() as tape2:
      tape2.watch(q)
      y = transformed_target_log_prob_fn(q)
    grads = tape2.gradient(y, q)
  hvp = tape1.gradient(grads, q, output_gradients = w)
  return hvp

