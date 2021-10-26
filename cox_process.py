####
#Experiments with Cox process model
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
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import PreconditionedHamiltonianMonteCarlo
import pandas as pd
from tensorflow_probability.python.math import psd_kernels as tfpk

tfd = tfp.distributions
tfb = tfp.bijectors


tfd = tfp.distributions
tfb = tfp.bijectors


flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=3,
                     help="number of leapfrog steps")
flags.DEFINE_integer("grid_dims",
                     default=8,
                     help="dimension of grid")
flags.DEFINE_integer("num_results",
                     default=1000,
                     help="number of MCMC steps")
flags.DEFINE_integer("num_burnin_steps",
                     default=0,
                     help="number of MCMC burnin steps")
flags.DEFINE_float("learning_rate",
                     default=0.0001,
                     help="learning rate for all optimizers")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'LogisticRegression'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("id",
                     default=0,
                     help="id of run for average results")
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
                   default = 1000,
                   help = "value for clipping gradients")
flags.DEFINE_bool('biased_accept_grads',
                  default = True,
                  help="whether to use biased gradients of the log accept terms"
                       "for the gradients")
flags.DEFINE_float("lipschitz_threshold_jvp",
                     default=0.99,
                     help="lipschitz threshold for clipping jpv")


FLAGS = flags.FLAGS


if True:
  FLAGS(sys.argv)

  tf.random.set_seed(FLAGS.id)
  np.random.seed(FLAGS.id)
  dtype = tf.float64

  grid_dims = FLAGS.grid_dims
  locations = np.stack(np.meshgrid(np.arange(grid_dims), np.arange(grid_dims)),
                       -1).reshape((-1, 2))
  locations = tf.cast(locations, dtype)
  # normalise locations to [0,1]x[0,1] grid
  locations = locations / grid_dims
  # generate true data as in Girolami (2011)
  true_length_scale = tf.constant(1. / 33, dtype = dtype)
  true_amplitude = tf.cast(tf.math.sqrt(1.91), dtype = dtype)
  true_mean_log_intensity = tf.cast(tf.math.log(126.), dtype = dtype) - .5 * true_amplitude ** 2
  m = 1. / (grid_dims ** 2)

  true_kernel = tfd.GaussianProcess(
    # pylint: disable=g-long-lambda
    mean_fn = lambda x: tf.broadcast_to(tf.expand_dims(true_mean_log_intensity, -1),
                                        true_mean_log_intensity.shape + [grid_dims ** 2]),
    kernel = tfpk.MaternOneHalf(
      amplitude = true_amplitude + 1e-6,
      length_scale = tf.cast(grid_dims, dtype) * true_length_scale + 1e-6
    ),
    index_points = locations,
    jitter = 1e-5)
  kernel_covariance = true_kernel.covariance()
  kernel_precision = tf.linalg.inv(kernel_covariance)
  kernel_precision_factor = tf.linalg.cholesky(kernel_precision)
  prior_dist = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
    loc = true_mean_log_intensity,
    precision = tf.linalg.LinearOperatorFullMatrix(kernel_precision),
    precision_factor = tf.linalg.LinearOperatorFullMatrix(kernel_precision_factor)
  )
  true_latent_field = prior_dist.sample()
  simulated_data = tfd.Poisson(
    rate = m * tf.math.exp(true_latent_field)).sample()
  reshaped_simulated_data = tf.reshape(simulated_data, [grid_dims, grid_dims])
  reshaped_true_latent_field = tf.reshape(true_latent_field, [grid_dims, grid_dims])


  def target(x):
    log_prior = prior_dist._log_prob_unnormalized(x)
    log_likelihood = tf.reduce_sum(tfd.Poisson(
      rate = m * tf.math.exp(x)).log_prob(simulated_data), -1)
    return log_prior + log_likelihood


  dims = grid_dims ** 2
  init_positions = np.random.normal(
    size = [FLAGS.num_chains, dims])
  init_positions = tf.constant(init_positions/100., dtype = dtype)
  target(init_positions)

  def make_pre_cond_fn(params):
    pre_cond_tril = params[0]
    pre_cond_operator = tf.linalg.LinearOperatorLowerTriangular(tril = pre_cond_tril)
    momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
      loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
      precision_factor = pre_cond_operator
    )
    return pre_cond_operator, momentum_distribution

  # params for preconditinong
  pre_cond_scale_diag_init = tfd.Normal(
    loc = 0., scale = .1).sample(dims) * tf.math.pow(tf.cast(dims, tf.float32), -1 / 3)
  pre_cond_tril = tf.Variable(
    initial_value = tf.cast(pre_cond_scale_diag_init * tf.eye(dims), dtype))
  pre_cond_params = [pre_cond_tril]


  #save flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path = os.path.join(FLAGS.model_dir,
                      'steps__{}_dim__{}_id__{}'.format(
                        flags.FLAGS.num_leapfrog_steps,
                        flags.FLAGS.grid_dims, flags.FLAGS.id))

  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()




  def penalty_fn(evs):
    delta_1 = .75
    delta_2 = 1.75
    return tf.nest.map_structure(
      lambda ev:  tf.where(abs(ev)<delta_2,
                           tf.where(abs(ev)<delta_1,
                                    tf.zeros_like(ev),
                                    (abs(ev)-delta_1)**2),
                           (delta_2-delta_1)**2+(delta_2-delta_1)*(abs(ev)-delta_2)
                           ),
      evs
    )

  def compute_ess(samples, dt):
    ess = tfp.mcmc.effective_sample_size(samples)
    min_ess_per_sec = tf.reduce_min(tf.reduce_mean(ess, 0)) / dt
    min_ess = tf.reduce_min(tf.reduce_mean(ess, 0))
    median_ess = np.median(ess)
    median_ess_per_sec = median_ess / dt
    return ess, median_ess, min_ess, min_ess_per_sec, median_ess_per_sec


  #trace fn (params/grads are saved in the kernel results for each
  #parallel chain
  def trace_fn(states, previous_kernel_results):
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
      previous_kernel_results.accepted_results.target_log_prob
    )


  precond_hmc = AdaptivePreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn = target,
        make_pre_cond_fn = make_pre_cond_fn,
        params = pre_cond_params,
        step_size = 1.,
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

  @tf.function(autograph=False)
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
      params_hmc_adapt,\
      grads_hmc_adapt,\
      target_log_probs_hmc_adapt) = results_hmc_adapt
  #results = run_precond_hmc_chain()
  time_hmc_adapt = time.time() - start_time_hmc_adapt

  ess_hmc_adapt, median_ess_hmc_adapt, min_ess_hmc_adapt, \
  min_ess_per_sec_hmc_adapt, median_ess_per_sec_hmc_adapt\
    = compute_ess(samples_hmc_adapt, time_hmc_adapt)
  pre_cond_operator, momentum_distribution = make_pre_cond_fn(pre_cond_params)
  C_adapt = pre_cond_operator.to_dense().numpy()
  inv_M_adapt = np.matmul(C_adapt,C_adapt.transpose())


  #save results

  np.savetxt(os.path.join(path, 'inv_M_adapt.csv'), inv_M_adapt, delimiter=",")
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


  learned_precond_hmc = PreconditionedHamiltonianMonteCarlo(
    target_log_prob_fn = target,
    step_size = precond_hmc._impl.inner_kernel._step_size.numpy(),
    num_leapfrog_steps = FLAGS.num_leapfrog_steps,
    momentum_distribution = momentum_distribution
  )

  @tf.function(autograph=False)
  def run_learned_precond_hmc_chain():
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=FLAGS.num_results,
      num_burnin_steps=FLAGS.num_burnin_steps,
      current_state=samples_hmc_adapt[-1],
      kernel=learned_precond_hmc,
      num_steps_between_results=0,
      return_final_kernel_results=False)

    return samples, kernel_results

  start_time_hmc_learned = time.time()
  samples_hmc_learned, kernel_results_hmc_learned \
    = run_learned_precond_hmc_chain()
  time_hmc_learned = time.time() - start_time_hmc_learned


  #Compute ESS statistics
  ess_hmc_learned, median_ess_hmc_learned, min_ess_hmc_learned,\
  min_ess_per_sec_hmc_learned, median_ess_per_sec_hmc_learned\
    = compute_ess(samples_hmc_learned, time_hmc_learned)
  #ESS for second moments
  ess_square_hmc_learned, median_ess_square_hmc_learned, min_ess_square_hmc_learned, \
  min_ess_square_per_sec_hmc_learned, median_ess_square_per_sec_hmc_learned \
      = compute_ess(tf.math.square(samples_hmc_learned), time_hmc_learned)
  #ESS for target log prob observable
  ess_target_hmc_learned, median_ess_target_hmc_learned, min_ess_target_hmc_learned, \
  min_ess_target_per_sec_hmc_learned, median_ess_target_per_sec_hmc_learned \
      = compute_ess(target(samples_hmc_learned), time_hmc_learned)

  #is_accepted_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_hmc_learned.inner_results.is_accepted,
  #                                     dtype = tf.float32))
  is_accepted_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_hmc_learned.is_accepted,
    dtype = tf.float32))
  target_log_probs_hmc_learned = kernel_results_hmc_learned.accepted_results.target_log_prob

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
  np.savetxt(os.path.join(path, 'ess_target_hmc_learned.csv'), ess_target_hmc_learned, delimiter=",")
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


  #####
  #Compare with NUTS
  #####

  nuts_sampler = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
    step_size = 0.1)

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
    = compute_ess(samples_nuts_adapt, time_nuts_adapt)

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
    target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
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
    = compute_ess(samples_nuts_learned, time_nuts_learned)
  #ESS for second moments
  ess_square_nuts_learned, median_ess_square_nuts_learned, min_ess_square_nuts_learned, \
  min_ess_square_per_sec_nuts_learned, median_ess_square_per_sec_nuts_learned \
      = compute_ess(tf.math.square(samples_nuts_learned), time_nuts_learned)
  #ESS for target log prob observable
  ess_target_nuts_learned, median_ess_target_nuts_learned, min_ess_target_nuts_learned, \
  min_ess_target_per_sec_nuts_learned, median_ess_target_per_sec_nuts_learned \
      = compute_ess(target(tf.cast(samples_nuts_learned, dtype)), time_nuts_learned)

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
  #np.save(os.path.join(path, 'samples_nuts_learned.npy'), samples_nuts_learned)
  #save ESS for second moment
  np.savetxt(os.path.join(path, 'ess_square_nuts_learned.csv'), ess_square_nuts_learned, delimiter=",")
  np.savetxt(os.path.join(path, 'min_ess_square_per_sec_nuts_learned.csv'), [min_ess_square_per_sec_nuts_learned], delimiter=",")
  np.savetxt(os.path.join(path, 'min_ess_square_nuts_learned.csv'), [min_ess_square_nuts_learned], delimiter=",")
  np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_square_per_sec_nuts_learned.csv'), [median_ess_square_per_sec_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned], delimiter=",")
  #save ESS for target observable
  np.savetxt(os.path.join(path, 'ess_target_nuts_learned.csv'), ess_target_nuts_learned, delimiter=",")
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