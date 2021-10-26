####
#Experiments with stochastic volatility model
####

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import os
from absl import flags, app
import sys
import matplotlib.pyplot as plt
from adaptive_linear_preconditioned_hmc import AdaptivePreconditionedHamiltonianMonteCarlo
import seaborn as sns
from inference_gym import using_tensorflow as gym
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import prefer_static
from inference_gym.targets import vectorized_stochastic_volatility
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import PreconditionedHamiltonianMonteCarlo
from upper_tridiag_operator import LinearOperatorUpperTridiag
import pandas as pd
tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32


flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=2,
                     help="number of leapfrog steps")
flags.DEFINE_string("data_set",
                     default="sp500_small",
                     help="name of dataset")
flags.DEFINE_integer("num_results",
                     default=20,
                     help="number of MCMC steps")
flags.DEFINE_string('pre_cond',
                    default='tridiag',
                    help="Pre-conditioner (diag or tridiag).")
flags.DEFINE_integer("num_burnin_steps",
                     default=0,
                     help="number of MCMC burnin steps")
flags.DEFINE_float("learning_rate",
                     default=0.05,
                     help="learning rate for precond parameters")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'StochVol'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("seed",
                     default=1,
                     help="seed")
flags.DEFINE_float("opt_acceptance_rate",
                     default=0.65,
                     help="targetet optimal acceptance rate")
flags.DEFINE_integer("num_exact_trace_terms",
                     default=3,
                     help="number of fixed (non-random) trace terms used")
flags.DEFINE_float("num_trace_terms_probs",
                   default = .5,
                   help = "parameter of the geometric distribution for the number of trace terms"
                          "used (on top of num_exact_trace_terms)")
flags.DEFINE_float("penalty_threshold",
                     default=0.95,
                     help="threshold where penalty starts")
flags.DEFINE_float("lipschitz_threshold_jvp",
                     default=0.99,
                     help="lipschitz threshold for clipping jpv")
flags.DEFINE_float("beta_min",
                   default = .001,
                   help = "minimum entropy weight")
flags.DEFINE_float("beta_max",
                   default = 100.,
                   help = "maximum entropy weight")
flags.DEFINE_float("beta_learning_rate",
                   default = .1,
                   help = "learning rate for entropy weight")
flags.DEFINE_float("clip_grad_value",
                   default = 100,
                   help = "value for clipping gradients")
flags.DEFINE_bool('biased_accept_grads',
                  default = True,
                  help="whether to use biased gradients of the log accept terms"
                       "for the gradients")
flags.DEFINE_string(
    'adaptation',
    default='grad',
    help="Adaptation type ('grad' or 'dual' or 'nuts'.")

FLAGS = flags.FLAGS


if True:
  FLAGS(sys.argv)
#def main(argv):
#  del argv  # unused
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  #######
  #helper functions for sampling unconstrained variables
  #from tfp
  #######

  # Transform target functions copied from tfp transformed kernel implementation
  def make_log_det_jacobian_fn(bijector, direction):
    """Makes a function which applies a list of Bijectors' `log_det_jacobian`s."""
    if not mcmc_util.is_list_like(bijector):
      bijector = [bijector]
    attr = '{}_log_det_jacobian'.format(direction)

    def fn(state_parts, event_ndims):
      return [
        getattr(b, attr)(sp, event_ndims = e)
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
    log_prob_fn, bijector, direction, enable_bijector_caching = True):
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


  ###########
  #Define target distribution depending on the used data set
  ###########
  if FLAGS.data_set in ['sp500']:
    base_model = vectorized_stochastic_volatility.VectorizedStochasticVolatilitySP500()

  elif FLAGS.data_set in ['sp500_small']:
    base_model = vectorized_stochastic_volatility.VectorizedStochasticVolatilitySP500Small()


  vec_model = gym.targets.VectorModel(base_model)
  target = make_transformed_log_prob(
      vec_model.unnormalized_log_prob, vec_model._default_event_space_bijector,
      'forward')

  dims = vec_model.event_shape[0]


  #save flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path = os.path.join(FLAGS.model_dir,
                    'data_set__{}_precond__{}_steps__{}_id__{}'.format(
                      flags.FLAGS.data_set, flags.FLAGS.pre_cond,
                      flags.FLAGS.num_leapfrog_steps, flags.FLAGS.seed))


  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags_'+FLAGS.adaptation+'.txt'), "w")
  flag_file.write(s)
  flag_file.close()


  def compute_cross_chain_ess(samples, dt):
    ess = tfp.mcmc.effective_sample_size(samples, cross_chain_dims = 1)
    min_ess_per_sec = tf.reduce_min(ess) / dt
    min_ess = tf.reduce_min(ess)
    median_ess = np.median(ess)
    median_ess_per_sec = median_ess / dt
    return ess, median_ess, min_ess, min_ess_per_sec, median_ess_per_sec

  def trace_fn_learned(states, previous_kernel_results):
    return (
        previous_kernel_results.is_accepted,
        previous_kernel_results.accepted_results.target_log_prob)


  #####
  #initialise parameters for the mass matrix
  #####
  pre_cond_scale_diag_init = tfd.Normal(
      loc = 100. * tf.math.pow(1.+tf.cast(FLAGS.num_leapfrog_steps, tf.float32), 1 / 3),
      scale = tf.math.sqrt(5.)).sample() * tf.ones([dims])
  init_positions = tfd.Normal(
      loc = 1., scale = tf.math.sqrt(.01)).sample([FLAGS.num_chains, dims])

  if FLAGS.adaptation == 'grad':

    pre_cond_scale_diag = tf.Variable(
      initial_value = tf.cast(pre_cond_scale_diag_init * tf.ones([dims]), dtype))
    if FLAGS.pre_cond == 'tridiag':
      pre_cond_scale_sub_diag_init = tfd.Normal(
        loc = 0., scale = 1.).sample(dims)
      pre_cond_scale_sub_diag = tf.Variable(initial_value = pre_cond_scale_sub_diag_init)
      pre_cond_params = [pre_cond_scale_diag, pre_cond_scale_sub_diag]
    elif FLAGS.pre_cond == 'diag':
      pre_cond_params = [pre_cond_scale_diag]


    def make_pre_cond_fn(params):
      pre_cond_scale_diag = params[0]
      #large learning rates can be unstable
      if FLAGS.pre_cond == 'diag':
        pre_cond_operator = tf.linalg.LinearOperatorInversion(
          tf.linalg.LinearOperatorDiag(diag = pre_cond_scale_diag))
      elif FLAGS.pre_cond =='tridiag':
        pre_cond_scale_sub_diag = params[1]
        pre_cond_scale_sub_diag = tf.tile(tf.expand_dims(pre_cond_scale_sub_diag,0),[FLAGS.num_chains,1])
        pre_cond_scale_diag = tf.tile(tf.expand_dims(pre_cond_scale_diag,0),[FLAGS.num_chains,1])
        pre_cond_operator = tf.linalg.LinearOperatorInversion(LinearOperatorUpperTridiag(
          diagonal = pre_cond_scale_diag, super_diagonal = pre_cond_scale_sub_diag))

      momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
        loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
        precision_factor = pre_cond_operator
      )

      return pre_cond_operator, momentum_distribution

    #trace fn (params/grads are saved in the kernel results for each
    #parallel chain
    def trace_fn(states, previous_kernel_results):
      if FLAGS.pre_cond == 'tridiag':
        return (
        previous_kernel_results.is_accepted,
        previous_kernel_results.log_accept_ratio,
        previous_kernel_results.proposed_results.mcmc_loss,
        previous_kernel_results.proposed_results.entropy_weight,
        previous_kernel_results.proposed_results.proposal_log_prob,
        previous_kernel_results.proposed_results.eigenvalue_estimate,
        previous_kernel_results.proposed_results.log_det_jac_transformation,
        previous_kernel_results.proposed_results.trace_residual_log_det,
        tf.concat([z[0] for z in previous_kernel_results.proposed_results.params], 0),
        tf.concat([z[0] for z in previous_kernel_results.proposed_results.grads], 0),
        previous_kernel_results.accepted_results.target_log_prob,
        previous_kernel_results.proposed_results.target_log_prob,
        previous_kernel_results.proposed_results.log_acceptance_correction
      )
      elif FLAGS.pre_cond == 'diag':
        return (
          previous_kernel_results.is_accepted,
          previous_kernel_results.log_accept_ratio,
          previous_kernel_results.proposed_results.mcmc_loss,
          previous_kernel_results.proposed_results.entropy_weight,
          previous_kernel_results.proposed_results.proposal_log_prob,
          previous_kernel_results.proposed_results.eigenvalue_estimate,
          previous_kernel_results.proposed_results.log_det_jac_transformation,
          previous_kernel_results.proposed_results.trace_residual_log_det,
          [z[0] for z in previous_kernel_results.proposed_results.params],
          [z[0] for z in previous_kernel_results.proposed_results.grads],
          previous_kernel_results.accepted_results.target_log_prob,
          previous_kernel_results.proposed_results.target_log_prob,
          previous_kernel_results.proposed_results.log_acceptance_correction
        )


    def penalty_fn(evs):
      delta_1 = FLAGS.penalty_threshold
      delta_2 = FLAGS.penalty_threshold + 1.
      return tf.nest.map_structure(
        lambda ev:  tf.where(abs(ev)<delta_2,
                             tf.where(abs(ev)<delta_1,
                                      0.,
                                      (abs(ev)-delta_1)**2),
                             (delta_2-delta_1)**2+(delta_2-delta_1)*(abs(ev)-delta_2)
                             ),
        evs
      )


    #lower momentum in Adam improves stability
    optimzer = tf.keras.optimizers.Adam(learning_rate = FLAGS.learning_rate, beta_1 = .0)

    precond_hmc = AdaptivePreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn = target,
        make_pre_cond_fn = make_pre_cond_fn,
        params = pre_cond_params,
        step_size = 1.,
        optimizer = optimzer,
        num_leapfrog_steps = FLAGS.num_leapfrog_steps,
        learning_rate = FLAGS.learning_rate,
        opt_acceptance_rate = FLAGS.opt_acceptance_rate,
        lipschitz_threshold_jvp = FLAGS.lipschitz_threshold_jvp,
        num_exact_trace_terms = FLAGS.num_exact_trace_terms,
        num_trace_terms_probs = FLAGS.num_trace_terms_probs,
        beta_learning_rate = FLAGS.beta_learning_rate,
        beta_min = FLAGS.beta_min,
        beta_max = FLAGS.beta_max,
        clip_grad_value = FLAGS.clip_grad_value,
        biased_accept_grads = FLAGS.biased_accept_grads,
        penalty_fn = penalty_fn
    )



    @tf.function
    def run_precond_hmc_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=tf.cast(init_positions, dtype),
        kernel=precond_hmc,
        trace_fn = trace_fn,
        return_final_kernel_results=False)

      return samples, kernel_results



    start_time_hmc_adapt = time.time()
    samples_hmc_adapt, results_hmc_adapt = run_precond_hmc_chain()
    (is_accepted_hmc_adapt,\
        log_accept_ratio_hmc_adapt,\
        mcmc_loss_hmc_adapt,\
        entropy_weight_hmc_adapt,\
        proposal_log_prob_hmc_adapt,\
        eigenvalue_estimate_hmc_adapt,\
        log_det_jac_transformation_hmc_adapt,\
        trace_residual_log_det_hmc_adapt,\
        params_hmc_adapt, \
        grads_hmc_adapt, \
        target_log_probs_hmc_adapt,\
        proposed_target_log_probs_hmc_adapt,\
        proposed_target_correction_hmc_adapt)= results_hmc_adapt

    time_hmc_adapt = time.time() - start_time_hmc_adapt
    ess_hmc_adapt, median_ess_hmc_adapt, min_ess_hmc_adapt, \
    min_ess_per_sec_hmc_adapt, median_ess_per_sec_hmc_adapt\
      = compute_cross_chain_ess(samples_hmc_adapt, time_hmc_adapt)
    pre_cond_operator, momentum_distribution = make_pre_cond_fn(pre_cond_params)
    if FLAGS.pre_cond == 'tridiag':
      C_adapt = pre_cond_operator.to_dense().numpy()[0]
      A_adapt = pre_cond_operator.inverse().to_dense().numpy()[0].transpose()
    elif FLAGS.pre_cond == 'diag':
      C_adapt = pre_cond_operator.to_dense().numpy()
      A_adapt = pre_cond_operator.inverse().to_dense().numpy().transpose()


    ############
    #save results
    ############
    inv_M_adapt = np.matmul(C_adapt,C_adapt.transpose())
    M_adapt = np.matmul(A_adapt, A_adapt.transpose())
    np.savetxt(os.path.join(path, 'ess_hmc_adapt.csv'), ess_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_hmc_adapt.csv'), [min_ess_per_sec_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_hmc_adapt.csv'), [min_ess_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'time_hmc_adapt.csv'), [time_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_hmc_adapt.csv'), [min_ess_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'proposal_log_prob_hmc_adapt.csv'), proposal_log_prob_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_hmc_adapt.csv'), np.mean(is_accepted_hmc_adapt,0), delimiter=",")
    np.savetxt(os.path.join(path, 'eigenvalue_estimate_hmc_adapt.csv'), eigenvalue_estimate_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'trace_residual_log_det_hmc_adapt.csv'), trace_residual_log_det_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_hmc_adapt.csv'), target_log_probs_hmc_adapt,
               delimiter = ",")

    plt.plot(tf.reduce_mean(proposal_log_prob_hmc_adapt, -1))
    plt.ylabel('proposal_log_prob')
    plt.savefig(os.path.join(path, 'proposal_log_prob.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    plt.plot(target_log_probs_hmc_adapt)
    plt.ylabel('target_log_prob')
    plt.savefig(os.path.join(path, 'target_log_probs_hmc_adapt.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    if FLAGS.pre_cond == 'diag':
      diag_param = params_hmc_adapt[0][:,:dims].numpy().reshape([FLAGS.num_results, -1])
      plt.plot(diag_param)
      plt.ylabel('diagonal params')
      plt.savefig(os.path.join(path, 'diag_param.png'), bbox_inches = 'tight')
      plt.show()
      plt.close()
      np.savetxt(os.path.join(path, 'diag_param.csv'), diag_param[-1], delimiter=",")

    if FLAGS.pre_cond == 'tridiag':
      diag_param = params_hmc_adapt[:, :dims].numpy().reshape([FLAGS.num_results, -1])
      plt.plot(diag_param)
      plt.ylabel('diagonal params')
      plt.savefig(os.path.join(path, 'diag_param.png'), bbox_inches = 'tight')
      plt.show()
      plt.close()
      np.savetxt(os.path.join(path, 'diag_param.csv'), diag_param[-1], delimiter = ",")

      subdiag_param = params_hmc_adapt[:, dims:].numpy().reshape([FLAGS.num_results, -1])
      plt.plot(subdiag_param)
      plt.ylabel('subdiagonal params')
      plt.savefig(os.path.join(path, 'subdiag_params.png'), bbox_inches = 'tight')
      plt.show()
      plt.close()
      np.savetxt(os.path.join(path, 'subdiag_params.csv'), subdiag_param[-1], delimiter=",")

    plt.plot(tf.reduce_mean(trace_residual_log_det_hmc_adapt, -1))
    plt.xlabel('iteration')
    plt.ylabel('trace_residual_log_det')
    plt.savefig(os.path.join(path, 'trace_residual_log_det.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    accept_ratio = pd.DataFrame(tf.reduce_mean(tf.minimum(1., tf.math.exp(
      log_accept_ratio_hmc_adapt)), -1))
    plt.plot(accept_ratio.rolling(50).mean())
    plt.xlabel('iteration')
    plt.ylabel('rolling_accept_ratio')
    plt.savefig(os.path.join(path, 'rolling_accept_ratio.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    plt.plot(tf.reduce_mean(eigenvalue_estimate_hmc_adapt, -1), label= 'mean')
    plt.plot(tf.reduce_min(eigenvalue_estimate_hmc_adapt, -1), label= 'min')
    plt.plot(tf.reduce_max(eigenvalue_estimate_hmc_adapt, -1), label = 'max')
    plt.xlabel('iteration')
    plt.legend()
    plt.ylabel('eigenvalue_estimate')
    plt.savefig(os.path.join(path, 'eigenvalue_estimate.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    sns.heatmap(inv_M_adapt)
    plt.savefig(os.path.join(path, 'inv_M_adapt.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    sns.heatmap(M_adapt)
    plt.savefig(os.path.join(path, 'M_adapt.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    # R hat diagnostics
    r_hat_weights_adapt = tfp.mcmc.potential_scale_reduction(
      samples_hmc_adapt, independent_chain_ndims = 1, split_chains = True)
    np.savetxt(os.path.join(path, 'r_hat_weights_adapt.csv'), r_hat_weights_adapt, delimiter = ",")

    plt.figure(figsize = (8, 4))
    plt.hist(r_hat_weights_adapt)
    plt.title("R-hat weights adaptation", size = 14)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'r_hat_weights_adapt.png'), bbox_inches = 'tight')
    plt.close()


    ###########
    #Run HMC with adapted mass matrix
    ###########

    learned_precond_hmc = PreconditionedHamiltonianMonteCarlo(
      target_log_prob_fn = target,
      step_size = precond_hmc._impl.inner_kernel._step_size.numpy(),
      num_leapfrog_steps = FLAGS.num_leapfrog_steps,
      momentum_distribution = momentum_distribution
    )

    @tf.function
    def run_learned_precond_hmc_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=samples_hmc_adapt[-1],
        kernel=learned_precond_hmc,
        num_steps_between_results=0,
        trace_fn = trace_fn_learned,
        return_final_kernel_results=False)

      return samples, kernel_results

    start_time_hmc_learned = time.time()
    samples_hmc_learned, kernel_results_hmc_learned \
      = run_learned_precond_hmc_chain()
    time_hmc_learned = time.time() - start_time_hmc_learned


    #Compute ESS statistics
    ess_hmc_learned, median_ess_hmc_learned, min_ess_hmc_learned,\
    min_ess_per_sec_hmc_learned, median_ess_per_sec_hmc_learned\
      = compute_cross_chain_ess(samples_hmc_learned, time_hmc_learned)
    #ESS for second moments
    ess_square_hmc_learned, median_ess_square_hmc_learned, min_ess_square_hmc_learned, \
    min_ess_square_per_sec_hmc_learned, median_ess_square_per_sec_hmc_learned \
        = compute_cross_chain_ess(tf.math.square(samples_hmc_learned), time_hmc_learned)
    #ESS for target log prob observable
    ess_target_hmc_learned, median_ess_target_hmc_learned, min_ess_target_hmc_learned, \
    min_ess_target_per_sec_hmc_learned, median_ess_target_per_sec_hmc_learned \
        = compute_cross_chain_ess(target(samples_hmc_learned), time_hmc_learned)

    is_accepted_hmc_learned, target_log_probs_hmc_learned = kernel_results_hmc_learned
    is_accepted_hmc_learned = tf.reduce_mean(tf.cast(is_accepted_hmc_learned,
      dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_hmc_learned.csv'), ess_hmc_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_hmc_learned.csv'), [min_ess_per_sec_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_hmc_learned.csv'), [min_ess_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_hmc_learned.csv'), [median_ess_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_per_sec_hmc_learned.csv'), [median_ess_per_sec_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_hmc_learned.csv'), [median_ess_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'time_hmc_learned.csv'), [time_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_hmc_learned.csv'), [is_accepted_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_hmc_learned.csv'), target_log_probs_hmc_learned, delimiter = ",")
    #np.save(os.path.join(path, 'samples_hmc_learned.npy'), samples_hmc_learned)
    #save ESS for second moment
    np.savetxt(os.path.join(path, 'ess_square_hmc_learned.csv'), ess_square_hmc_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_square_per_sec_hmc_learned.csv'), [min_ess_square_per_sec_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_square_hmc_learned.csv'), [min_ess_square_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_square_hmc_learned.csv'), [median_ess_square_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_square_per_sec_hmc_learned.csv'), [median_ess_square_per_sec_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_square_hmc_learned.csv'), [median_ess_square_hmc_learned], delimiter=",")
    #save ESS for target observable
    np.savetxt(os.path.join(path, 'ess_target_hmc_learned.csv'), [ess_target_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_per_sec_hmc_learned.csv'), [min_ess_target_per_sec_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_hmc_learned.csv'), [min_ess_target_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_target_hmc_learned.csv'), [median_ess_target_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_target_per_sec_hmc_learned.csv'), [median_ess_target_per_sec_hmc_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_target_hmc_learned.csv'), [median_ess_target_hmc_learned], delimiter=",")

    plt.plot(target_log_probs_hmc_learned)
    plt.ylabel('target_log_probs_hmc_learned')
    plt.savefig(os.path.join(path, 'target_log_probs_hmc_learned.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    # R hat diagnostics
    r_hat_weights_learned = tfp.mcmc.potential_scale_reduction(
      samples_hmc_learned, independent_chain_ndims = 1, split_chains = True)
    np.savetxt(os.path.join(path, 'r_hat_weights_learned.csv'), r_hat_weights_learned, delimiter = ",")

    plt.figure(figsize = (8, 4))
    plt.hist(r_hat_weights_learned)
    plt.title("R-hat weights", size = 14)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'r_hat_weights_learned.png'), bbox_inches = 'tight')
    plt.close()

  elif FLAGS.adaptation == 'dual':

    #####
    #Compare with HMC using dual adaptation
    #####
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn = target,
      step_size = .01,
      num_leapfrog_steps = FLAGS.num_leapfrog_steps
    )
    adaptive_dual_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel = hmc,
      num_adaptation_steps = FLAGS.num_results,
      target_accept_prob = tf.cast(FLAGS.opt_acceptance_rate, dtype))

    @tf.function
    def run_adaptive_dual_hmc_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=init_positions,
        kernel=adaptive_dual_sampler,
        num_steps_between_results=0,
        return_final_kernel_results=False)

      return samples, kernel_results

    start_time_dual_hmc_adapt = time.time()
    samples_dual_hmc_adapt, kernel_results_dual_hmc_adapt \
      = run_adaptive_dual_hmc_chain()
    time_dual_hmc_adapt = time.time() - start_time_dual_hmc_adapt

    ess_dual_hmc_adapt, median_ess_dual_hmc_adapt, min_ess_dual_hmc_adapt, \
    min_ess_per_sec_dual_hmc_adapt, median_ess_per_sec_dual_hmc_adapt\
      = compute_cross_chain_ess(samples_dual_hmc_adapt, time_dual_hmc_adapt)

    is_accepted_dual_hmc_adapt = tf.reduce_mean(
      tf.cast(kernel_results_dual_hmc_adapt.inner_results.is_accepted,
      dtype = tf.float32))
    target_log_probs_dual_hmc_adapt = kernel_results_dual_hmc_adapt.inner_results.accepted_results.target_log_prob
    dual_hmc_adapted_step_sizes = kernel_results_dual_hmc_adapt.new_step_size

    np.savetxt(os.path.join(path, 'ess_dual_hmc_adapt.csv'), ess_dual_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_dual_hmc_adapt.csv'), [min_ess_per_sec_dual_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_dual_hmc_adapt.csv'), [min_ess_dual_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'time_dual_hmc_adapt.csv'), [time_dual_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_dual_hmc_adapt.csv'), [is_accepted_dual_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_dual_hmc_adapt.csv'), [median_ess_dual_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_dual_hmc_adapt.csv'), target_log_probs_dual_hmc_adapt, delimiter = ",")

    plt.plot(target_log_probs_dual_hmc_adapt)
    plt.ylabel('target_log_probs_dual_hmc_adapt')
    plt.savefig(os.path.join(path, 'target_log_probs_dual_hmc_adapt.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()


    ####
    #Run HMC with adapted step size
    ###
    learned_dual_precond_hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn = target,
      step_size = tf.nest.map_structure(lambda a: a[-1], dual_hmc_adapted_step_sizes),
      num_leapfrog_steps = FLAGS.num_leapfrog_steps)

    @tf.function(autograph=False)
    def run_learned_dual_precond_hmc_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=samples_dual_hmc_adapt[-1],
        kernel=learned_dual_precond_hmc,
        num_steps_between_results=0,
        return_final_kernel_results=False,
        trace_fn = trace_fn_learned)

      return samples, kernel_results

    start_time_dual_hmc_learned = time.time()
    samples_dual_hmc_learned, kernel_results_dual_hmc_learned \
      = run_learned_dual_precond_hmc_chain()
    time_dual_hmc_learned = time.time() - start_time_dual_hmc_learned

    #Compute ESS statistics
    ess_dual_hmc_learned, median_ess_dual_hmc_learned, min_ess_dual_hmc_learned, min_ess_per_sec_dual_hmc_learned, _\
      = compute_cross_chain_ess(samples_dual_hmc_learned, time_dual_hmc_learned)

    is_accepted_dual_hmc_learned, target_log_probs_dual_hmc_learned = kernel_results_dual_hmc_learned
    is_accepted_dual_hmc_learned = tf.reduce_mean(tf.cast(is_accepted_dual_hmc_learned,
      dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_dual_hmc_learned.csv'), ess_dual_hmc_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_dual_hmc_learned.csv'), [min_ess_per_sec_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_dual_hmc_learned.csv'), [min_ess_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'time_dual_hmc_learned.csv'), [time_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_dual_hmc_learned.csv'), [is_accepted_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_dual_hmc_learned.csv'), [median_ess_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_dual_hmc_learned.csv'), target_log_probs_dual_hmc_learned, delimiter = ",")

    plt.plot(target_log_probs_dual_hmc_learned)
    plt.ylabel('target_log_probs_dual_hmc_learned')
    plt.savefig(os.path.join(path, 'target_log_probs_dual_hmc_learned.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    #ESS for target log prob observable
    ess_target_dual_hmc_learned, median_ess_target_dual_hmc_learned, min_ess_target_dual_hmc_learned, \
    min_ess_target_per_sec_dual_hmc_learned, median_ess_target_per_sec_dual_hmc_learned \
        = compute_cross_chain_ess(target(samples_dual_hmc_learned), time_dual_hmc_learned)
    np.savetxt(os.path.join(path, 'ess_target_dual_hmc_learned.csv'), [ess_target_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_per_sec_dual_hmc_learned.csv'), [min_ess_target_per_sec_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_dual_hmc_learned.csv'), [min_ess_target_dual_hmc_learned], delimiter=",")


    # R hat diagnostics
    r_hat_weights_dual_learned = tfp.mcmc.potential_scale_reduction(
      samples_dual_hmc_learned, independent_chain_ndims = 1, split_chains = True)
    np.savetxt(os.path.join(path, 'r_hat_weights_dual_learned.csv'), r_hat_weights_dual_learned, delimiter = ",")

    plt.figure(figsize = (8, 4))
    plt.hist(r_hat_weights_dual_learned)
    plt.title("R-hat weights", size = 14)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'r_hat_weights_dual_learned.png'), bbox_inches = 'tight')
    plt.close()


  elif FLAGS.adaptation == 'nuts':

    #####
    #Compare with NUTS
    #####

    nuts_sampler = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn = target,
      step_size = 0.005)

    adaptive_nuts_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel=nuts_sampler,
      num_adaptation_steps=FLAGS.num_results,
      target_accept_prob=FLAGS.opt_acceptance_rate)

    num_results_nuts = FLAGS.num_results


    @tf.function(autograph=False)
    def run_adaptive_nuts_chain():
      # Run the chain (with burn-in).
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results_nuts,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=tf.cast(init_positions, tf.float32),
        kernel=adaptive_nuts_sampler,
        num_steps_between_results=0,
        return_final_kernel_results=False,)

      return samples, kernel_results


    start_time_nuts_adapt = time.time()
    samples_nuts_adapt,  kernel_results_nuts_adapt  = run_adaptive_nuts_chain()
    time_nuts_adapt = time.time() - start_time_nuts_adapt


    ess_nuts_adapt, median_ess_nuts_adapt, min_ess_nuts_adapt, \
    min_ess_per_sec_nuts_adapt, median_ess_per_sec_nuts_adapt\
      = compute_cross_chain_ess(samples_nuts_adapt, time_nuts_adapt)

    nuts_adapted_step_sizes = kernel_results_nuts_adapt.new_step_size
    is_accepted_nuts_adapt = tf.reduce_mean(tf.cast(kernel_results_nuts_adapt.inner_results.is_accepted,
                                         dtype = tf.float32))
    target_log_probs_nuts_adapt = kernel_results_nuts_adapt.inner_results.target_log_prob

    np.savetxt(os.path.join(path, 'ess_nuts_adapt.csv'), ess_nuts_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'time_nuts_adapt.csv'), [time_nuts_adapt], delimiter = ",")
    np.savetxt(os.path.join(path, 'nuts_adapted_step_sizes.csv'), nuts_adapted_step_sizes, delimiter = ",")
    np.savetxt(os.path.join(path, 'is_accepted_nuts_adapt.csv'), [is_accepted_nuts_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'steps_nuts_adapt.csv'), kernel_results_nuts_adapt.inner_results.leapfrogs_taken,
               delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_nuts_adapt.csv'), target_log_probs_nuts_adapt,
               delimiter=",")
    plt.plot(target_log_probs_nuts_adapt)
    plt.ylabel('target_log_probs_nuts_adapt')
    plt.savefig(os.path.join(path, 'target_log_probs_nuts_adapt.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()


    learned_nuts_sampler = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn = target,
      step_size = tf.nest.map_structure(lambda a: a[-1], nuts_adapted_step_sizes))


    @tf.function(autograph=False)
    def run_learned_nuts_chain():
      # Run the chain (with burn-in).
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results_nuts,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=tf.cast(samples_nuts_adapt[-1], tf.float32),
        kernel=learned_nuts_sampler,
        num_steps_between_results=0,
        return_final_kernel_results=False,)
      return samples, kernel_results



    start_time_nuts_learned = time.time()
    samples_nuts_learned, kernel_results_nuts_learned = run_learned_nuts_chain()
    time_nuts_learned = time.time() - start_time_nuts_learned

    #ESS calculations
    ess_nuts_learned, median_ess_nuts_learned, min_ess_nuts_learned, \
    min_ess_per_sec_nuts_learned, median_ess_per_sec_nuts_learned \
      = compute_cross_chain_ess(samples_nuts_learned, time_nuts_learned)
    #ESS for second moments
    ess_square_nuts_learned, median_ess_square_nuts_learned, min_ess_square_nuts_learned, \
    min_ess_square_per_sec_nuts_learned, median_ess_square_per_sec_nuts_learned \
        = compute_cross_chain_ess(tf.math.square(samples_nuts_learned), time_nuts_learned)
    #ESS for target log prob observable
    ess_target_nuts_learned, median_ess_target_nuts_learned, min_ess_target_nuts_learned, \
    min_ess_target_per_sec_nuts_learned, median_ess_target_per_sec_nuts_learned \
        = compute_cross_chain_ess(target(samples_nuts_learned), time_nuts_learned)

    is_accepted_nuts_learned = tf.reduce_mean(tf.cast(kernel_results_nuts_learned.is_accepted,
                                         dtype = tf.float32))
    target_log_probs_nuts_learned = kernel_results_nuts_learned.target_log_prob

    np.savetxt(os.path.join(path, 'ess_nuts_learned.csv'), ess_nuts_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_nuts_learned.csv'), [min_ess_per_sec_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_nuts_learned.csv'), [min_ess_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_nuts_learned.csv'), [median_ess_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_per_sec_nuts_learned.csv'), [median_ess_per_sec_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'time_nuts_learned.csv'), [time_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_nuts_learned.csv'), [is_accepted_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'steps_nuts_learned.csv'), kernel_results_nuts_learned.leapfrogs_taken, delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_nuts_learned.csv'), target_log_probs_nuts_learned, delimiter = ",")
    #save ESS for second moment
    np.savetxt(os.path.join(path, 'ess_square_nuts_learned.csv'), ess_square_nuts_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_square_per_sec_nuts_learned.csv'), [min_ess_square_per_sec_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_square_nuts_learned.csv'), [min_ess_square_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_square_per_sec_nuts_learned.csv'), [median_ess_square_per_sec_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned], delimiter=",")
    #save ESS for target observable
    np.savetxt(os.path.join(path, 'ess_target_nuts_learned.csv'), [ess_target_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_per_sec_nuts_learned.csv'), [min_ess_target_per_sec_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_target_nuts_learned.csv'), [min_ess_target_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'median_ess_target_nuts_learned.csv'), [median_ess_target_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_target_per_sec_nuts_learned.csv'), [median_ess_target_per_sec_nuts_learned], delimiter = ",")
    np.savetxt(os.path.join(path, 'median_ess_target_nuts_learned.csv'), [median_ess_target_nuts_learned], delimiter=",")


    plt.plot(target_log_probs_nuts_learned)
    plt.ylabel('target_log_probs_nuts_learned')
    plt.savefig(os.path.join(path, 'target_log_probs_nuts_learned.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    # R hat diagnostics
    r_hat_weights_nuts_learned = tfp.mcmc.potential_scale_reduction(
      samples_nuts_learned, independent_chain_ndims = 1, split_chains = True)
    np.savetxt(os.path.join(path, 'r_hat_weights_nuts_learned.csv'), r_hat_weights_nuts_learned, delimiter = ",")

    plt.figure(figsize = (8, 4))
    plt.hist(r_hat_weights_nuts_learned)
    plt.title("R-hat weights", size = 14)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'r_hat_weights_nuts_learned.png'), bbox_inches = 'tight')
    plt.close()


#if __name__ == '__main__':
#  app.run(main)