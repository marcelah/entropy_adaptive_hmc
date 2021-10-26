"""Entropy-based adaptiv Hamiltonian Monte Carlo algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import metropolis_hastings
import leapfrog_integrator_trajectory as leapfrog_impl
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.util.seed_stream import SeedStream
import collections
from tensorflow_probability.python.distributions import Uniform


__all__ = [
    'AdaptivePreconditionedHamiltonianMonteCarlo',
]

class AdaptiveKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('AdaptiveKernelResults',
            [
            'log_acceptance_correction',
            'target_log_prob',
            'grads_target_log_prob',
            'initial_momentum',
            'final_momentum',
            'step_size',
            'num_leapfrog_steps',
            'seed',
            'mcmc_loss',
            'params',
            'grads',
             ])):
  __slots__ = ()

class AdaptiveEsjdPreconditionedHamiltonianMonteCarlo(hmc.HamiltonianMonteCarlo):
  """Runs Hamiltonian Monte Carlo with a non-identity mass matrix."""

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               make_pre_cond_fn,
               params,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               seed=None,
               store_parameters_in_results=False,
               name=None,
               learning_rate = .005,
               clip_grad_value = 1000.,
               l2hmc = False,
               optimizer = None
               ):

    self._seed_stream = SeedStream(seed, salt='hmc')
    uhmc_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    mh_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedEsjdAdaptivePreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            make_pre_cond_fn=make_pre_cond_fn,
            params=params,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
            learning_rate = learning_rate,
            clip_grad_value = clip_grad_value,
            l2hmc = l2hmc,
            optimizer = optimizer,
            **uhmc_kwargs),
        **mh_kwargs)
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters['step_size_update_fn'] = step_size_update_fn
    self._parameters['seed'] = seed


class UncalibratedEsjdAdaptivePreconditionedHamiltonianMonteCarlo(
    hmc.UncalibratedHamiltonianMonteCarlo):

  @mcmc_util.set_doc(hmc.UncalibratedHamiltonianMonteCarlo.__init__)
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               make_pre_cond_fn,
               params,
               learning_rate,
               clip_grad_value,
               l2hmc,
               optimizer,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               name=None,
               ):
    super(UncalibratedEsjdAdaptivePreconditionedHamiltonianMonteCarlo, self).__init__(
        target_log_prob_fn,
        step_size,
        num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        store_parameters_in_results=store_parameters_in_results,
        name=name
    )
    self._parameters['make_pre_cond_fn'] = make_pre_cond_fn
    self._parameters['params'] = params
    if optimizer is None:
      self._optimizer_pre_cond = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
      self._optimizer_pre_cond = optimizer
    self._learning_rate = learning_rate
    self._step_size = tf.Variable(initial_value = step_size, trainable = False, name ='step_size')
    self._clip_grad_value = clip_grad_value
    self._l2hmc = l2hmc
    self._scale = tf.Variable(initial_value = 1., trainable = True)



  @property
  def make_pre_cond_fn(self):
    return self._parameters['make_pre_cond_fn']

  @property
  def params(self):
    return self._parameters['params']

  @mcmc_util.set_doc(hmc.HamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'hmc', 'one_step')):

      with tf.GradientTape(persistent = False) as tape1:

        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps
        pre_cond_operator, momentum_distribution = self.make_pre_cond_fn(self.params)

        [
            current_state_parts,
            step_sizes,
            momentum_distribution,
            pre_cond_operator,
            current_target_log_prob,
            current_target_log_prob_grad_parts,
        ] = _prepare_args(
            self.target_log_prob_fn,
            current_state,
            step_size,
            momentum_distribution,
            pre_cond_operator,
            previous_kernel_results.target_log_prob,
            previous_kernel_results.grads_target_log_prob,
            maybe_expand=True,
            state_gradients_are_stopped=self.state_gradients_are_stopped)

        seed = samplers.sanitize_seed(seed)
        seeds = samplers.split_seed(seed, n = len(current_state_parts))

        current_momentum_noise_parts = tf.nest.map_structure(
         lambda x,s: tf.cast(samplers.normal(shape=tf.shape(x), seed=s), x.dtype),
          current_state_parts, seeds)
        tape1.watch(current_momentum_noise_parts)

        current_momentum_parts = tf.nest.map_structure(
         lambda C,v: C.solvevec(v, adjoint = True),
          pre_cond_operator, current_momentum_noise_parts)

        momentum_log_prob = getattr(momentum_distribution,
                                  '_log_prob_unnormalized',
                                  momentum_distribution.log_prob)
        kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)

        leapfrog_kinetic_energy_fn = kinetic_energy_fn

        integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
            self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

        [
            next_momentum_parts,
            next_state_parts,
            next_target_log_prob,
            next_target_log_prob_grad_parts,
            momentum_parts_array,
            target_log_prob_grad_parts_array_non_stopped,
            state_parts_array
        ] = integrator(
            current_momentum_parts,
            current_state_parts,
            target=current_target_log_prob,
            target_grad_parts=current_target_log_prob_grad_parts,
            kinetic_energy_fn=leapfrog_kinetic_energy_fn,
            )
        if self.state_gradients_are_stopped:
          next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

        def maybe_flatten(x):
          return x if mcmc_util.is_list_like(current_state) else x[0]

        #linear/langevin part for proposal log prob
        dims = tf.nest.map_structure(
          lambda x: tf.cast(tf.shape(x)[1:], x.dtype), current_state_parts)


        log_acceptance_correction = _compute_log_acceptance_correction(
                kinetic_energy_fn, current_momentum_parts,
                next_momentum_parts)
        #compute log-acceptance rates
        to_sum = [next_target_log_prob,
                  -previous_kernel_results.target_log_prob,
                  log_acceptance_correction]
        log_accept_ratio = mcmc_util.safe_sum(
          to_sum, name = 'log_accept_ratio')
        accept_rate = tf.minimum(tf.ones([], dtype = log_accept_ratio.dtype), tf.math.exp(log_accept_ratio))


        #loss for optimizing
        #sample uniform rvs to check for acceptance (MC estimate for ESJD)
        #u = Uniform().sample(log_accept_ratio.shape)
        #accept_indicator = tf.cast(tf.math.log(u) < log_accept_ratio, log_accept_ratio.dtype)
        esjd = tf.nest.map_structure(lambda x,y: accept_rate*tf.linalg.norm(x-y, axis=-1),
                                     current_state_parts, next_state_parts)
        esjd_loss = - mcmc_util.safe_sum(esjd)
        if not self._l2hmc:
          mcmc_loss = esjd_loss
        else:
          scale = tf.cast(tf.stop_gradient(self._scale), esjd_loss.dtype)
          mcmc_loss =scale /(-esjd_loss) + esjd_loss/scale
          #exponential moving average of esjd as weight
          self._scale.assign(.99*self._scale + tf.cast(
            .01*(tf.reduce_mean(-esjd_loss)), tf.float32))
          tf.print('mcmc_loss', mcmc_loss)
          tf.print('esjd_loss', esjd_loss)
          tf.print(self._scale)


      grads = tape1.gradient(mcmc_loss, self.params)
      target_log_probs, _ = mcmc_util.maybe_call_fn_and_grads(
            self.target_log_prob_fn, tf.nest.map_structure(
          lambda x: tf.ensure_shape(x, [num_leapfrog_steps+1]+x.shape[1:]), state_parts_array)
        , None, None)


      #replace nans
      grads = tf.nest.map_structure(
        lambda g: tf.where(tf.math.is_nan(g), tf.zeros_like(g), g), grads)
      #clip grads
      grads = tf.nest.map_structure(lambda g: tf.clip_by_value(g, -self._clip_grad_value, self._clip_grad_value),
                                    grads)

      #update params
      self._optimizer_pre_cond.apply_gradients(zip(grads, self.params))

      # decrease step_size if nan loss
      h_new = tf.cond(tf.math.is_finite(tf.reduce_sum(mcmc_loss)),
                      lambda: self._step_size,
                      lambda: self._step_size * (1 - self._learning_rate)
                      )
      self._step_size.assign(h_new)








    new_kernel_results = previous_kernel_results._replace(
      log_acceptance_correction = log_acceptance_correction,
      target_log_prob = next_target_log_prob,
      grads_target_log_prob = next_target_log_prob_grad_parts,
      initial_momentum = current_momentum_parts,
      final_momentum = next_momentum_parts,
      seed = seed,
      mcmc_loss = mcmc_loss,
      grads = tf.nest.map_structure(lambda c: tf.broadcast_to(c, [current_state.shape[0]] + c.shape),
                                    grads),
      params = tf.nest.map_structure(lambda c: tf.broadcast_to(c, [current_state.shape[0]]+c.shape),
                                     self.params)
      )

    return maybe_flatten(next_state_parts), new_kernel_results


  def bootstrap_results(self, init_state):
    hmc_results = super().bootstrap_results(init_state)
    num_chains = init_state.shape[0]
    zero_grads = tf.nest.map_structure(lambda x: tf.broadcast_to(x,[num_chains]+x.shape),
                                       self.params)
    kernel_results = AdaptiveKernelResults(
      log_acceptance_correction = hmc_results.log_acceptance_correction,
      target_log_prob = hmc_results.target_log_prob,
      grads_target_log_prob = hmc_results.grads_target_log_prob,
      initial_momentum = hmc_results.initial_momentum,
      final_momentum = hmc_results.final_momentum,
      step_size = hmc_results.step_size,
      num_leapfrog_steps = hmc_results.num_leapfrog_steps,
      seed = hmc_results.seed,
      mcmc_loss = hmc_results.log_acceptance_correction,
      grads = zero_grads,
      params = tf.nest.map_structure(lambda c: tf.broadcast_to(c, [num_chains]+c.shape),
                                     self.params),
    )

    return kernel_results


def _compute_log_acceptance_correction(kinetic_energy_fn,
                                       current_momentums,
                                       proposed_momentums,
                                       name=None):
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


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  momentum_distribution,
                  pre_cond_operator,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=target_log_prob.dtype, name='step_size')


  # The momentum will get "maybe listified" to zip with the state parts,
  # and this step makes sure that the momentum distribution will have the
  # same "maybe listified" underlying shape.
  if not mcmc_util.is_list_like(momentum_distribution.dtype):
    momentum_distribution = jds.JointDistributionSequential(
        [momentum_distribution])

  if not mcmc_util.is_list_like(pre_cond_operator.dtype):
    pre_cond_operator = [pre_cond_operator]

  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      momentum_distribution,
      pre_cond_operator,
      target_log_prob,
      grads_target_log_prob,
  ]
