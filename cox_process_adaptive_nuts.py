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
import seaborn as sns
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import PreconditionedHamiltonianMonteCarlo
import pandas as pd
from tensorflow_probability.python.math import psd_kernels as tfpk

tfd = tfp.distributions
tfb = tfp.bijectors

tfd = tfp.distributions
tfb = tfp.bijectors

flags.DEFINE_string(
    'adaptation',
    default='nuts_diag',
    help="Adaptation type.")
flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("depth",
                     default=5,
                     help="tree depth")
flags.DEFINE_integer("grid_dims",
                     default=8,
                     help="dimension of grid")
flags.DEFINE_integer("num_results",
                     default=1000,
                     help="number of MCMC steps")
flags.DEFINE_integer("num_burnin_steps",
                     default=0,
                     help="number of MCMC burnin steps")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'CoxProcess'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("id",
                     default=0,
                     help="id of run for average results")
flags.DEFINE_float("opt_acceptance_rate",
                     default=0.65,
                     help="targetet optimal acceptance rate")


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

  #for nuts with windows adaptation
  joint = tfd.JointDistributionNamed(dict(
    w = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
      loc = tf.cast(true_mean_log_intensity, tf.float32),
      precision = tf.linalg.LinearOperatorFullMatrix(tf.cast(kernel_precision, tf.float32)),
      precision_factor = tf.linalg.LinearOperatorFullMatrix(
        tf.cast(kernel_precision_factor, tf.float32))),
    y = lambda w: tfd.Independent(tfd.Poisson(
      rate = m * tf.math.exp(w)),
      reinterpreted_batch_ndims = 1)
  ))
  target_dist = tfp.experimental.distributions.JointDistributionPinned(joint, y = tf.cast(simulated_data, tf.float32))

  dims = grid_dims ** 2
  init_positions = np.random.normal(
    size = [FLAGS.num_chains, dims])

  #save flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path = os.path.join(FLAGS.model_dir,
                      'adapt__{}_depth__{}_dim__{}_id__{}'.format(
                        flags.FLAGS.adaptation,
                        flags.FLAGS.depth,
                        flags.FLAGS.grid_dims, flags.FLAGS.id))

  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()



  def compute_ess(samples, dt):
    ess = tfp.mcmc.effective_sample_size(samples)
    min_ess_per_sec = tf.reduce_min(tf.reduce_mean(ess, 0)) / dt
    min_ess = tf.reduce_min(tf.reduce_mean(ess, 0))
    median_ess = np.median(ess)
    median_ess_per_sec = median_ess / dt
    return ess, median_ess, min_ess, min_ess_per_sec, median_ess_per_sec


  #####
  #Compare with NUTS
  #####

  if FLAGS.adaptation == 'nuts_windows':
    init_positions = {'w': tf.constant(init_positions / 100., dtype = tf.float32)}
    windowed_adaptive_nuts = tfp.experimental.mcmc.windowed_adaptive_nuts(
      n_draws = 1,
      joint_dist = target_dist,
      n_chains = FLAGS.num_chains,
      num_adaptation_steps = FLAGS.num_results,
      current_state = init_positions,
      init_step_size = .1,
      max_tree_depth = FLAGS.depth,
      return_final_kernel_results = True,
      discard_tuning = False,
      dual_averaging_kwargs = {'step_count_smoothing' : 100,
                               'target_accept_prob': FLAGS.opt_acceptance_rate,
                               'reduce_fn': tfp.math.reduce_log_harmonic_mean_exp,
                               #'decay_rate': 0.5,
                               #'exploration_shrinkage': 0.2
                                }
    )

    np.savetxt(os.path.join(path, 'nuts_adapted_variance.csv'),
               windowed_adaptive_nuts.final_kernel_results.momentum_distribution.variance()[0],
               delimiter = ",")
    np.savetxt(os.path.join(path, 'nuts_windows_adapted_step_sizes.csv'),
               windowed_adaptive_nuts.trace['step_size'], delimiter = ",")
    np.savetxt(os.path.join(path, 'is_accepted_nuts_windows_adapt.csv'),
               [np.mean(windowed_adaptive_nuts.trace['is_accepted'])], delimiter=",")
    np.savetxt(os.path.join(path, 'steps_nuts_windows_adapt.csv'),
               windowed_adaptive_nuts.trace['n_steps'], delimiter=",")


    ######
    #manual step size setting !!!!!!!!!
    #####

    learned_nuts_sampler = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
      step_size = tf.nest.map_structure(
        lambda x: .4*tf.ones_like(x), windowed_adaptive_nuts.final_kernel_results.step_size),
      #step_size = tf.nest.map_structure(lambda a: a[-1], kernel_results_nuts_adapt.new_step_size),
      target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
      max_tree_depth = FLAGS.depth,
      momentum_distribution = windowed_adaptive_nuts.final_kernel_results.momentum_distribution
    )

    # learned_nuts_sampler = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
    #   step_size = windowed_adaptive_nuts.final_kernel_results.step_size,
    #   #step_size = tf.nest.map_structure(lambda a: a[-1], kernel_results_nuts_adapt.new_step_size),
    #   target_log_prob_fn = target_dist.unnormalized_log_prob,
    #   max_tree_depth = 5,
    #   momentum_distribution = windowed_adaptive_nuts.final_kernel_results.momentum_distribution
    # )



    @tf.function(autograph=False)
    def run_learned_nuts_chain():
      # Run the chain (with burn-in).
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=windowed_adaptive_nuts.all_states['w'][-1],
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


  else:
    tfpe = tfp.experimental
    init_positions = tf.constant(init_positions / 100., dtype = tf.float32)


    @tf.function
    def first_window(num_adaptation_steps):
      prior_variance = tfpe.stats.RunningVariance.from_stats(
        num_samples = 10., mean = tf.zeros([FLAGS.num_chains, dims]),
        variance = tf.ones([FLAGS.num_chains, dims]))

      nuts_kernel = tfpe.mcmc.PreconditionedNoUTurnSampler(
        target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
        # lambda x: target(tf.expand_dims(x, 0)),
        step_size = .1
      )
      adapting_kernel = tfpe.mcmc.DiagonalMassMatrixAdaptation(
        tfp.mcmc.SimpleStepSizeAdaptation(nuts_kernel,
                                          num_adaptation_steps = num_adaptation_steps),
        initial_running_variance = prior_variance)
      draws, _, fkr = tfp.mcmc.sample_chain(num_adaptation_steps,
                                            init_positions,
                                            kernel = adapting_kernel,
                                            return_final_kernel_results = True,
                                            trace_fn = None)
      return draws, fkr


    draws, fkr = first_window(100)
    new_variance = fkr.running_variance
    new_step = fkr.inner_results.inner_results.step_size
    new_variance = [tfpe.stats.RunningVariance.from_stats(
      num_samples = rv.num_samples / 10.,
      mean = rv.mean,
      variance = rv.variance()) for rv in new_variance]


    @tf.function
    def second_window(num_adaptation_steps, prior_variance, step_size, init):
      nuts_kernel = tfpe.mcmc.PreconditionedNoUTurnSampler(
        target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
        step_size = step_size)
      adapting_kernel = tfpe.mcmc.DiagonalMassMatrixAdaptation(
        tfp.mcmc.SimpleStepSizeAdaptation(nuts_kernel,
                                          num_adaptation_steps = num_adaptation_steps),
        initial_running_variance = prior_variance)
      draws, _, fkr = tfp.mcmc.sample_chain(num_adaptation_steps, init,
                                            kernel = adapting_kernel,
                                            return_final_kernel_results = True,
                                            trace_fn = None)
      return draws, fkr


    for _ in range(5):
      draws, fkr = second_window(tf.constant(FLAGS.num_results // 5), new_variance, new_step, draws[-1])

      new_variance = fkr.running_variance
      new_step = fkr.inner_results.inner_results.step_size

      print(f'''variance: {tfp.stats.variance(draws, 0)}
      mean: {tf.reduce_mean(draws, 0)}\n''')

      # We now go and downweight  our reliance on the previous estimate
      new_variance = [tfpe.stats.RunningVariance.from_stats(
        num_samples = rv.num_samples / 10.,
        mean = rv.mean,
        variance = rv.variance()) for rv in new_variance]

  np.savetxt(os.path.join(path, 'nuts_adapted_variance.csv'),
               fkr.inner_results.inner_results.momentum_distribution.variance()[0],
               delimiter = ",")

  learned_nuts_sampler = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
    step_size = new_step,
    target_log_prob_fn = lambda x: tf.cast(target(tf.cast(x, dtype)), tf.float32),
    max_tree_depth = FLAGS.depth,
    momentum_distribution = fkr.inner_results.inner_results.momentum_distribution
  )


  @tf.function(autograph = False)
  def run_learned_nuts_chain():
    # Run the chain (with burn-in).
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results = FLAGS.num_results,
      num_burnin_steps = FLAGS.num_burnin_steps,
      current_state = draws[-1],
      kernel = learned_nuts_sampler,
      num_steps_between_results = 0,
      return_final_kernel_results = False, )
    return samples, kernel_results


  start_time_nuts_learned = time.time()
  samples_nuts_learned, kernel_results_nuts_learned = run_learned_nuts_chain()
  time_nuts_learned = time.time() - start_time_nuts_learned

  # ESS calculations
  ess_nuts_learned, median_ess_nuts_learned, min_ess_nuts_learned, \
  min_ess_per_sec_nuts_learned, median_ess_per_sec_nuts_learned \
    = compute_ess(samples_nuts_learned, time_nuts_learned)
  # ESS for second moments
  ess_square_nuts_learned, median_ess_square_nuts_learned, min_ess_square_nuts_learned, \
  min_ess_square_per_sec_nuts_learned, median_ess_square_per_sec_nuts_learned \
    = compute_ess(tf.math.square(samples_nuts_learned), time_nuts_learned)
  # ESS for target log prob observable
  ess_target_nuts_learned, median_ess_target_nuts_learned, min_ess_target_nuts_learned, \
  min_ess_target_per_sec_nuts_learned, median_ess_target_per_sec_nuts_learned \
    = compute_ess(target(tf.cast(samples_nuts_learned, dtype)), time_nuts_learned)

  is_accepted_nuts_learned = tf.reduce_mean(tf.cast(kernel_results_nuts_learned.is_accepted,
                                                    dtype = tf.float32))
  target_log_probs_nuts_learned = kernel_results_nuts_learned.target_log_prob

  np.savetxt(os.path.join(path, 'ess_nuts_learned.csv'), ess_nuts_learned, delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_per_sec_nuts_learned.csv'), [min_ess_per_sec_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_nuts_learned.csv'), [min_ess_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_nuts_learned.csv'), [median_ess_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_per_sec_nuts_learned.csv'), [median_ess_per_sec_nuts_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'time_nuts_learned.csv'), [time_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'is_accepted_nuts_learned.csv'), [is_accepted_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'steps_nuts_learned.csv'), kernel_results_nuts_learned.leapfrogs_taken,
             delimiter = ",")
  np.savetxt(os.path.join(path, 'target_log_probs_nuts_learned.csv'), target_log_probs_nuts_learned, delimiter = ",")
  # np.save(os.path.join(path, 'samples_nuts_learned.npy'), samples_nuts_learned)
  # save ESS for second moment
  np.savetxt(os.path.join(path, 'ess_square_nuts_learned.csv'), ess_square_nuts_learned, delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_square_per_sec_nuts_learned.csv'), [min_ess_square_per_sec_nuts_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_square_nuts_learned.csv'), [min_ess_square_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_square_per_sec_nuts_learned.csv'),
             [median_ess_square_per_sec_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_square_nuts_learned.csv'), [median_ess_square_nuts_learned],
             delimiter = ",")
  # save ESS for target observable
  np.savetxt(os.path.join(path, 'ess_target_nuts_learned.csv'), ess_target_nuts_learned, delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_target_per_sec_nuts_learned.csv'), [min_ess_target_per_sec_nuts_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_target_nuts_learned.csv'), [min_ess_target_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_target_nuts_learned.csv'), [median_ess_target_nuts_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_target_per_sec_nuts_learned.csv'),
             [median_ess_target_per_sec_nuts_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_target_nuts_learned.csv'), [median_ess_target_nuts_learned],
             delimiter = ",")

  plt.plot(target_log_probs_nuts_learned)
  plt.ylabel('target_log_probs_nuts_learned')
  plt.savefig(os.path.join(path, 'target_log_probs_nuts_learned.png'), bbox_inches = 'tight')
  plt.show()
  plt.close()

