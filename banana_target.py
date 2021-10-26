import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import os
from absl import flags
import sys
import matplotlib.pyplot as plt
import pandas as pd
from adaptive_transformed_hmc import AdaptivePreconditionedHamiltonianMonteCarlo
from inference_gym import using_tensorflow as gym

tfd = tfp.distributions
tfb = tfp.bijectors

dtype = tf.float32

flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=2,
                     help="number of leapfrog steps")
flags.DEFINE_string('bijector',
                    default = 'real_nvp_sigmoid',
                    help="bijector type.")
flags.DEFINE_integer("dims",
                     default=2,
                     help="target dimensions")
flags.DEFINE_float("curvature",
                     default=0.1,
                     help="curvature of target")
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
    default=os.path.join(os.getcwd(),'BananaTarget'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("seed",
                     default=0,
                     help="seed")
flags.DEFINE_float("opt_acceptance_rate",
                     default=0.65,
                     help="targetet optimal acceptance rate")
flags.DEFINE_integer("num_exact_trace_terms",
                     default=2,
                     help="number of fixed (non-random) trace terms used")
flags.DEFINE_float("num_trace_terms_probs",
                   default = .5,
                   help = "parameter of the geometric distribution for the number of trace terms"
                          "used (on top of num_exact_trace_terms)")
flags.DEFINE_float("penalty_threshold",
                     default=0.95,
                     help="threshold where penalty starts")
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
FLAGS = flags.FLAGS






if True:
  FLAGS(sys.argv)
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  dims = FLAGS.dims
  target = gym.targets.Banana(
    ndims=dims, curvature = FLAGS.curvature
  ).unnormalized_log_prob

  #save flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path = os.path.join(FLAGS.model_dir,
                    'curvature__{}_bijector__{}_steps__{}_dims__{}_id__{}'.format(
                      flags.FLAGS.curvature, flags.FLAGS.bijector, flags.FLAGS.num_leapfrog_steps,
                      flags.FLAGS.dims, flags.FLAGS.seed))


  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()


  def make_sigmoid_real_nvp_bijector():
    hidden_size = 50
    shift_and_sigmoid_scale_fn1 = tfb.real_nvp_default_template(
      hidden_layers = [hidden_size, hidden_size], name = '1',
      activation = tf.nn.elu)
    shift_and_sigmoid_scale_fn2 = tfb.real_nvp_default_template(
      hidden_layers = [hidden_size, hidden_size], name = '2',
      activation = tf.nn.elu)
    def _bijector_fn1(x0, input_depth, **condition_kwargs):

      shift, sigmoid_scale = shift_and_sigmoid_scale_fn1(x0, input_depth,
                                                **condition_kwargs)
      bijectors = []
      bijectors.append(tfb.Shift(shift))
      low = .5
      high = 2.
      bijectors.append(tfb.Scale(
        scale = high * tf.sigmoid(sigmoid_scale) + low * tf.sigmoid(-sigmoid_scale)))
      return tfb.Chain(bijectors)

    def _bijector_fn2(x0, input_depth, **condition_kwargs):
      shift, sigmoid_scale = shift_and_sigmoid_scale_fn2(x0, input_depth,
                                                **condition_kwargs)
      bijectors = []
      bijectors.append(tfb.Shift(shift))
      low = .5
      high = 2.
      bijectors.append(tfb.Scale(
        scale = high * tf.sigmoid(sigmoid_scale) + low * tf.sigmoid(-sigmoid_scale)))
      return tfb.Chain(bijectors)

    bijector1 = tfb.RealNVP(
      num_masked = 1,
      bijector_fn = _bijector_fn1)
    permutation = list(reversed(range(dims)))
    swap_bijector = tfb.Permute(permutation)
    bijector2 = tfb.RealNVP(
      num_masked = dims-1,
      bijector_fn = _bijector_fn2)

    log_scale = tf.Variable(initial_value = -tf.ones([dims]))
    scale_bijector = tfp.bijectors.Scale(log_scale = log_scale)

    bijector = tfb.Chain([scale_bijector, swap_bijector, bijector2,
                          swap_bijector, bijector1])
    return bijector



  def make_linear_bijector():
    pre_cond_tril = tf.Variable(
      initial_value = .1 * np.ones([dims]) * tf.eye(dims))
    bijector_params = [pre_cond_tril]
    bijector = tfb.ScaleMatvecTriL(scale_tril = bijector_params[0])
    return bijector


  def compute_cross_chain_ess(samples, dt):
    ess = tfp.mcmc.effective_sample_size(samples, cross_chain_dims = 1)
    min_ess_per_sec = tf.reduce_min(ess) / dt
    min_ess = tf.reduce_min(ess)
    median_ess = np.median(ess)
    median_ess_per_sec = median_ess / dt
    return ess, median_ess, min_ess, min_ess_per_sec, median_ess_per_sec



  #trace fn (params/grads are saved in the kernel results for each
  #parallel chain
  def trace_fn(states, previous_kernel_results):
    return (
      previous_kernel_results.accepted_results.transformed_state,
      previous_kernel_results.is_accepted,
      previous_kernel_results.proposed_results.eigenvalue,
      previous_kernel_results.proposed_results.proposal_log_prob,
      previous_kernel_results.accepted_results.target_log_prob,
      previous_kernel_results.log_accept_ratio
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

  if FLAGS.bijector == 'real_nvp_sigmoid':
    bijector = make_sigmoid_real_nvp_bijector()
  elif FLAGS.bijector == 'linear':
    bijector = make_linear_bijector()

  precond_hmc = AdaptivePreconditionedHamiltonianMonteCarlo(
    target_log_prob_fn = target,
    bijector_fn = bijector,
    step_size = .1,
    num_leapfrog_steps = FLAGS.num_leapfrog_steps,
    learning_rate = FLAGS.learning_rate,
    opt_acceptance_rate = FLAGS.opt_acceptance_rate,
    num_exact_trace_terms = FLAGS.num_exact_trace_terms,
    num_trace_terms_probs = FLAGS.num_trace_terms_probs,
    beta_learning_rate = FLAGS.beta_learning_rate,
    beta_min = FLAGS.beta_min,
    beta_max = FLAGS.beta_max,
    clip_grad_value = FLAGS.clip_grad_value,
    biased_accept_grads = FLAGS.biased_accept_grads,
    penalty_fn = penalty_fn
  )

  transformed_init_positions = tf.random.normal([FLAGS.num_chains, dims])
  init_positions = bijector(transformed_init_positions)

  @tf.function
  def run_precond_hmc_chain():
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=FLAGS.num_results,
      num_burnin_steps=FLAGS.num_burnin_steps,
      current_state = init_positions,
      previous_kernel_results = precond_hmc.bootstrap_results(transformed_init_positions),
      kernel=precond_hmc,
      trace_fn = trace_fn,
      return_final_kernel_results=False,
      seed=FLAGS.seed)
    return samples, kernel_results




  start_time_hmc_adapt = time.time()
  samples_hmc_adapt, results_hmc_adapt = run_precond_hmc_chain()
  (transformed_samples_hmc_adapt, is_accepted_hmc_adapt, eigenvalue_estimate_hmc_adapt, \
   proposal_log_prob_hmc_adapt, transformed_target_log_probs_hmc_adapt, log_accept_ratio_hmc_adapt) = results_hmc_adapt
  #results = run_precond_hmc_chain()
  time_hmc_adapt = time.time() - start_time_hmc_adapt

  ess_hmc_adapt, median_ess_hmc_adapt, min_ess_hmc_adapt, \
  min_ess_per_sec_hmc_adapt, median_ess_per_sec_hmc_adapt\
    = compute_cross_chain_ess(samples_hmc_adapt, time_hmc_adapt)

  std_samples = (tf.math.reduce_std(samples_hmc_adapt[FLAGS.num_results//2:],axis=[0, 1]))
  std_transformed_samples = (tf.math.reduce_std(transformed_samples_hmc_adapt[FLAGS.num_results // 2:],
                                                axis = [0, 1]))

  target_log_probs_hmc_adapt = target(samples_hmc_adapt)

  #save results
  np.savetxt(os.path.join(path, 'ess_hmc_adapt.csv'), ess_hmc_adapt, delimiter=",")
  np.savetxt(os.path.join(path, 'min_ess_per_sec_hmc_adapt.csv'), [min_ess_per_sec_hmc_adapt], delimiter=",")
  np.savetxt(os.path.join(path, 'min_ess_hmc_adapt.csv'), [min_ess_hmc_adapt], delimiter=",")
  np.savetxt(os.path.join(path, 'time_hmc_adapt.csv'), [time_hmc_adapt], delimiter=",")
  np.savetxt(os.path.join(path, 'min_ess_hmc_adapt.csv'), [min_ess_hmc_adapt], delimiter=",")
  np.savetxt(os.path.join(path, 'proposal_log_prob_hmc_adapt.csv'), proposal_log_prob_hmc_adapt, delimiter=",")
  np.savetxt(os.path.join(path, 'is_accepted_hmc_adapt.csv'), np.mean(is_accepted_hmc_adapt,0), delimiter=",")
  np.savetxt(os.path.join(path, 'eigenvalue_estimate_hmc_adapt.csv'), eigenvalue_estimate_hmc_adapt, delimiter=",")
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
  plt.plot(transformed_target_log_probs_hmc_adapt)
  plt.ylabel('transformed_target_log_prob')
  plt.savefig(os.path.join(path, 'transformed_target_log_probs_hmc_adapt.png'), bbox_inches = 'tight')
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

  ######
  #Run learned sampler
  ######

  learned_precond_hmc =tfp.mcmc.TransformedTransitionKernel(
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn = target,
      step_size = precond_hmc._impl.inner_kernel._step_size.numpy(),
      num_leapfrog_steps = FLAGS.num_leapfrog_steps),
    bijector = bijector
  )

  @tf.function
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
    = compute_cross_chain_ess(samples_hmc_learned, time_hmc_learned)
  #ESS for second moments
  ess_square_hmc_learned, median_ess_square_hmc_learned, min_ess_square_hmc_learned, \
  min_ess_square_per_sec_hmc_learned, median_ess_square_per_sec_hmc_learned \
      = compute_cross_chain_ess(tf.math.square(samples_hmc_learned), time_hmc_learned)
  #ESS for target log prob observable
  ess_target_hmc_learned, median_ess_target_hmc_learned, min_ess_target_hmc_learned, \
  min_ess_target_per_sec_hmc_learned, median_ess_target_per_sec_hmc_learned \
      = compute_cross_chain_ess(target(samples_hmc_learned), time_hmc_learned)

  #is_accepted_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_hmc_learned.inner_results.is_accepted,
  #                                     dtype = tf.float32))
  is_accepted_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_hmc_learned.inner_results.is_accepted,
    dtype = tf.float32))
  target_log_probs_hmc_learned = target(samples_hmc_learned)

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



  #####
  # Compare with HMC using dual adaptation
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
      num_results = FLAGS.num_results,
      num_burnin_steps = FLAGS.num_burnin_steps,
      current_state = init_positions,
      kernel = adaptive_dual_sampler,
      num_steps_between_results = 0,
      return_final_kernel_results = False)

    return samples, kernel_results


  start_time_dual_hmc_adapt = time.time()
  samples_dual_hmc_adapt, kernel_results_dual_hmc_adapt \
    = run_adaptive_dual_hmc_chain()
  time_dual_hmc_adapt = time.time() - start_time_dual_hmc_adapt

  ess_dual_hmc_adapt, median_ess_dual_hmc_adapt, min_ess_dual_hmc_adapt, \
  min_ess_per_sec_dual_hmc_adapt, median_ess_per_sec_dual_hmc_adapt \
    = compute_cross_chain_ess(samples_dual_hmc_adapt, time_dual_hmc_adapt)

  is_accepted_dual_hmc_adapt = tf.reduce_mean(
    tf.cast(kernel_results_dual_hmc_adapt.inner_results.is_accepted,
            dtype = tf.float32))
  target_log_probs_dual_hmc_adapt = kernel_results_dual_hmc_adapt.inner_results.accepted_results.target_log_prob
  dual_hmc_adapted_step_sizes = kernel_results_dual_hmc_adapt.new_step_size

  np.savetxt(os.path.join(path, 'ess_dual_hmc_adapt.csv'), ess_dual_hmc_adapt, delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_per_sec_dual_hmc_adapt.csv'), [min_ess_per_sec_dual_hmc_adapt],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_dual_hmc_adapt.csv'), [min_ess_dual_hmc_adapt], delimiter = ",")
  np.savetxt(os.path.join(path, 'time_dual_hmc_adapt.csv'), [time_dual_hmc_adapt], delimiter = ",")
  np.savetxt(os.path.join(path, 'is_accepted_dual_hmc_adapt.csv'), [is_accepted_dual_hmc_adapt], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_dual_hmc_adapt.csv'), [median_ess_dual_hmc_adapt], delimiter = ",")
  np.savetxt(os.path.join(path, 'target_log_probs_dual_hmc_adapt.csv'), target_log_probs_dual_hmc_adapt,
             delimiter = ",")

  plt.plot(target_log_probs_dual_hmc_adapt)
  plt.ylabel('target_log_probs_dual_hmc_adapt')
  plt.savefig(os.path.join(path, 'target_log_probs_dual_hmc_adapt.png'), bbox_inches = 'tight')
  plt.show()
  plt.close()

  ####
  # Run HMC with adapted step size
  ###
  learned_dual_precond_hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn = target,
    step_size = tf.nest.map_structure(lambda a: a[-1], dual_hmc_adapted_step_sizes),
    num_leapfrog_steps = FLAGS.num_leapfrog_steps)


  @tf.function(autograph = False)
  def run_learned_dual_precond_hmc_chain():
    samples, kernel_results = tfp.mcmc.sample_chain(
      num_results = FLAGS.num_results,
      num_burnin_steps = FLAGS.num_burnin_steps,
      current_state = samples_dual_hmc_adapt[-1],
      kernel = learned_dual_precond_hmc,
      num_steps_between_results = 0,
      return_final_kernel_results = False)

    return samples, kernel_results


  start_time_dual_hmc_learned = time.time()
  samples_dual_hmc_learned, kernel_results_dual_hmc_learned \
    = run_learned_dual_precond_hmc_chain()
  time_dual_hmc_learned = time.time() - start_time_dual_hmc_learned

  # Compute ESS statistics
  ess_dual_hmc_learned, median_ess_dual_hmc_learned, min_ess_dual_hmc_learned, min_ess_per_sec_dual_hmc_learned, _ \
    = compute_cross_chain_ess(samples_dual_hmc_learned, time_dual_hmc_learned)

  # is_accepted_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_hmc_learned.inner_results.is_accepted,
  #                                     dtype = tf.float32))
  is_accepted_dual_hmc_learned = tf.reduce_mean(tf.cast(kernel_results_dual_hmc_learned.is_accepted,
                                                        dtype = tf.float32))
  target_log_probs_dual_hmc_learned = kernel_results_dual_hmc_learned.accepted_results.target_log_prob

  np.savetxt(os.path.join(path, 'ess_dual_hmc_learned.csv'), ess_dual_hmc_learned, delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_per_sec_dual_hmc_learned.csv'), [min_ess_per_sec_dual_hmc_learned],
             delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_dual_hmc_learned.csv'), [min_ess_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'time_dual_hmc_learned.csv'), [time_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'is_accepted_dual_hmc_learned.csv'), [is_accepted_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'median_ess_dual_hmc_learned.csv'), [median_ess_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'target_log_probs_dual_hmc_learned.csv'), target_log_probs_dual_hmc_learned,
             delimiter = ",")
  # np.save(os.path.join(path, 'samples_hmc_learned.npy'), samples_hmc_learned)

  plt.plot(target_log_probs_dual_hmc_learned)
  plt.ylabel('target_log_probs_dual_hmc_learned')
  plt.savefig(os.path.join(path, 'target_log_probs_dual_hmc_learned.png'), bbox_inches = 'tight')
  plt.show()
  plt.close()

  # ESS for target log prob observable
  ess_target_dual_hmc_learned, median_ess_target_dual_hmc_learned, min_ess_target_dual_hmc_learned, \
  min_ess_target_per_sec_dual_hmc_learned, median_ess_target_per_sec_dual_hmc_learned \
    = compute_cross_chain_ess(target(samples_dual_hmc_learned), time_dual_hmc_learned)
  np.savetxt(os.path.join(path, 'ess_target_dual_hmc_learned.csv'), [ess_target_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_target_per_sec_dual_hmc_learned.csv'),
             [min_ess_target_per_sec_dual_hmc_learned], delimiter = ",")
  np.savetxt(os.path.join(path, 'min_ess_target_dual_hmc_learned.csv'), [min_ess_target_dual_hmc_learned],
             delimiter = ",")

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


  #####
  #Compare with NUTS
  #####

  nuts_sampler = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn = target,
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
  #np.save(os.path.join(path, 'samples_nuts_learned.npy'), samples_nuts_learned)
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