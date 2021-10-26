####
#Experiments with different Gaussian targets
####
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import os
from absl import flags, app
import sys
import matplotlib.pyplot as plt
from adaptive_esjd_linear_preconditioned_hmc import AdaptiveEsjdPreconditionedHamiltonianMonteCarlo
import seaborn as sns
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import PreconditionedHamiltonianMonteCarlo
import pandas as pd
tfd = tfp.distributions
tfb = tfp.bijectors


flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=1,
                     help="number of leapfrog steps")
flags.DEFINE_string(
    'adaptation',
    default='grad_esjd',
    help="Adaptation type ('grad_esjd' or 'grad_l2hmc' or 'dual' or nuts'.")
flags.DEFINE_string("cov_type",
                     default='GP',
                     help="covariance matrix. Must be from ['GP', 'log_space_independent',"
                          "'log_space_independent_easier', 'iid']")
flags.DEFINE_integer("dims",
                     default=31,
                     help="dimension of target")
flags.DEFINE_integer("num_results",
                     default=2000,
                     help="number of MCMC steps")
flags.DEFINE_integer("num_burnin_steps",
                     default=0,
                     help="number of MCMC burnin steps")
flags.DEFINE_float("learning_rate",
                     default=0.001,
                     help="learning rate for all optimizers")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'Gaussian'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("id",
                     default=0,
                     help="id of run for average results (used for seed)")
flags.DEFINE_float("clip_grad_value",
                   default = 1000,
                   help = "value for clipping gradients")


FLAGS = flags.FLAGS


if True:
  FLAGS(sys.argv)
  #def main(argv):
  # del argv  # unused
  tf.random.set_seed(FLAGS.id)
  np.random.seed(FLAGS.id)
  dtype = tf.float32

  if FLAGS.cov_type=='GP':
    #Construct GP covariance function
    dims = FLAGS.dims
    index_points = np.expand_dims(np.linspace(0., 4.,  dims), -1)
    gp_cov_matrix_=np.zeros([dims,dims])
    for i in range(0,dims):
      for j in range(0,dims):
        gp_cov_matrix_[i,j]=np.exp(-.5*(index_points[i]-index_points[j])**2/.16)
    for i in range(0,dims):
      gp_cov_matrix_[i,i]=gp_cov_matrix_[i,i]+.01

    target_dist = tfd.MultivariateNormalTriL(
      loc = tf.zeros(dims), scale_tril = tf.linalg.cholesky(tf.cast(gp_cov_matrix_, dtype)))
    target = target_dist.log_prob

    def hessian_potential(z):
      return tf.linalg.inv(tf.cast(gp_cov_matrix_, dtype))

    #eigenvalues of target and initialise initial samples and parameters
    eigenvalues_target, _ = tf.linalg.eigh(target_dist.covariance())
    smooth_param = 1/eigenvalues_target[0]
    init_positions = tf.random.normal([FLAGS.num_chains, dims])/tf.sqrt(smooth_param)

    #params for preconditinong
    #for esjd/l2hmc adaptation, using smaller values seems necessary
    #there is no adaptation if accept rates are zero (rounded)
    pre_cond_scale_diag_init = tfd.Normal(
      loc = 0., scale = .01).sample(dims)/FLAGS.num_leapfrog_steps
    pre_cond_tril = tf.Variable(
        initial_value =pre_cond_scale_diag_init * tf.eye(dims))
    pre_cond_params = [pre_cond_tril]

    def make_pre_cond_fn(params):
      pre_cond_tril = params[0]
      pre_cond_operator = tf.linalg.LinearOperatorLowerTriangular(tril = pre_cond_tril)
      momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
        loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
        precision_factor = pre_cond_operator
      )
      return pre_cond_operator, momentum_distribution

  elif FLAGS.cov_type == 'log_space_independent':

    dims = FLAGS.dims
    diags = np.exp(3*np.linspace(0., dims-1, dims) / (dims-1) *np.log(10))
    std = tf.Variable(diags, dtype=dtype, name='std')
    target_dist = tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims), scale_diag=std)
    target = target_dist.log_prob

    def hessian_potential(z):
      return tf.linalg.diag(1/std**2)

    smooth_param = tf.cast(1,dtype)
    init_positions = tf.random.normal([FLAGS.num_chains,dims])/tf.sqrt(smooth_param)

    #params for preconditinong
    pre_cond_scale_diag_init = tfd.Normal(
      loc = 0., scale = 1).sample(dims) * tf.math.pow(tf.cast(dims, tf.float32), -1 / 3)
    pre_cond_scale_diag = tf.Variable(
      initial_value = pre_cond_scale_diag_init * tf.ones([dims]))
    pre_cond_params = [pre_cond_scale_diag]

    def make_pre_cond_fn(params):
      pre_cond_scale_diag = params[0]
      pre_cond_operator = tf.linalg.LinearOperatorDiag(diag = pre_cond_scale_diag)
      momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
        loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
        precision_factor = pre_cond_operator
      )
      return pre_cond_operator, momentum_distribution


  elif FLAGS.cov_type == 'log_space_independent_easier':
    #log space scaling from 1 to 10 only
    dims = FLAGS.dims
    diags = np.exp(1*np.linspace(0., dims-1, dims) / (dims-1) *np.log(10))
    std = tf.Variable(diags, dtype=dtype, name='std')
    target_dist = tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims), scale_diag=std)
    target = target_dist.log_prob

    def hessian_potential(z):
      return tf.tile(tf.expand_dims(tf.linalg.diag(1/std**2),0), [z.shape[0],1,1])

    smooth_param = tf.cast(1,dtype)
    init_positions = tf.random.normal([FLAGS.num_chains,dims])/tf.sqrt(smooth_param)

    #params for preconditinong
    pre_cond_scale_diag_init = tfd.Normal(
      loc = 0., scale = 1).sample(dims) * tf.math.pow(tf.cast(dims, tf.float32), -1 / 3)
    pre_cond_scale_diag = tf.Variable(
      initial_value = pre_cond_scale_diag_init * tf.ones([dims]))
    pre_cond_params = [pre_cond_scale_diag]

    def make_pre_cond_fn(params):
      pre_cond_scale_diag = params[0]
      pre_cond_operator = tf.linalg.LinearOperatorDiag(diag = pre_cond_scale_diag)
      momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
        loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
        precision_factor = pre_cond_operator
      )
      return pre_cond_operator, momentum_distribution



  elif FLAGS.cov_type == 'iid':

    dims = FLAGS.dims
    target_dist = tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims), scale_diag=tf.ones(dims))
    target = target_dist.log_prob

    smooth_param = tf.cast(1,dtype)
    init_positions = tf.random.normal([FLAGS.num_chains,dims])/tf.sqrt(smooth_param)

    #params for preconditinong
    pre_cond_scale_diag_init = tfd.Normal(
      loc = 0., scale = 1).sample(dims) * tf.math.pow(tf.cast(dims, tf.float32), -1 / 3)
    pre_cond_scale_diag = tf.Variable(
      initial_value = pre_cond_scale_diag_init * tf.ones([dims]))

    pre_cond_params = [pre_cond_scale_diag]

    def make_pre_cond_fn(params):
      pre_cond_scale_diag = params[0]
      pre_cond_operator = tf.linalg.LinearOperatorDiag(diag = pre_cond_scale_diag)
      momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
        loc = tf.cast(tf.zeros([FLAGS.num_chains, dims]), dtype),
        precision_factor = pre_cond_operator
      )
      return pre_cond_operator, momentum_distribution




  #save flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path=os.path.join(FLAGS.model_dir,
                    'covariance__{}__adapt__{}_steps__{}_dims__{}_id__{}'.format(
                      flags.FLAGS.cov_type, flags.FLAGS.adaptation, flags.FLAGS.num_leapfrog_steps,
                      flags.FLAGS.dims, flags.FLAGS.id))
  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags_'+FLAGS.adaptation+'.txt'), "w")
  flag_file.write(s)
  flag_file.close()

  def compute_ess(samples, dt):
    ess = tfp.mcmc.effective_sample_size(samples)
    min_ess_per_sec = tf.reduce_min(tf.reduce_mean(ess, 0)) / dt
    min_ess = tf.reduce_min(tf.reduce_mean(ess, 0))
    mean_ess = np.mean(ess)
    return ess, mean_ess, min_ess, min_ess_per_sec



  def trace_fn_learned(states, previous_kernel_results):
    return (
        previous_kernel_results.is_accepted,
        previous_kernel_results.accepted_results.target_log_prob)


  def penalty_fn(evs):
    delta_1 = FLAGS.penalty_threshold
    delta_2 = FLAGS.penalty_threshold + 1.
    return tf.nest.map_structure(
      lambda ev: tf.where(abs(ev) < delta_2,
                          tf.where(abs(ev) < delta_1,
                                   0.,
                                   (abs(ev) - delta_1) ** 2),
                          (delta_2 - delta_1) ** 2 + (delta_2 - delta_1) * (abs(ev) - delta_2)
                          ),
      evs
    )

  #trace fn (params/grads are saved in the kernel results for each
  #parallel chain
  def trace_fn(states, previous_kernel_results):
    return (
      previous_kernel_results.is_accepted,
      previous_kernel_results.log_accept_ratio,
      previous_kernel_results.proposed_results.mcmc_loss,
      previous_kernel_results.proposed_results.params[0][0],
      previous_kernel_results.proposed_results.grads[0][0]
    )



  if FLAGS.adaptation in ['grad_esjd', 'grad_l2hmc']:

    precond_hmc = AdaptiveEsjdPreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn = target,
        make_pre_cond_fn = make_pre_cond_fn,
        params = pre_cond_params,
        step_size = 1.,
        num_leapfrog_steps = FLAGS.num_leapfrog_steps,
        learning_rate = FLAGS.learning_rate,
        clip_grad_value = FLAGS.clip_grad_value,
        l2hmc = True if FLAGS.adaptation=='grad_l2hmc' else False
    )

    @tf.function
    def run_precond_hmc_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=FLAGS.num_burnin_steps,
        current_state=tf.cast(init_positions, dtype),
        kernel=precond_hmc,
        num_steps_between_results=0,
        trace_fn = trace_fn,
        return_final_kernel_results=False)

      return samples, kernel_results


    start_time_hmc_adapt = time.time()
    samples_hmc_adapt, results_hmc_adapt = run_precond_hmc_chain()
    (is_accepted_hmc_adapt,\
        log_accept_ratio_hmc_adapt,\
        mcmc_loss_hmc_adapt,\
        params_hmc_adapt,\
        grads_hmc_adapt
     ) = results_hmc_adapt
    #results = run_precond_hmc_chain()
    time_hmc_adapt = time.time() - start_time_hmc_adapt
    np.savetxt(os.path.join(path, 'time_hmc_adapt.csv'), [time_hmc_adapt], delimiter=",")
    ess_hmc_adapt, mean_ess_hmc_adapt, min_ess_hmc_adapt, min_ess_per_sec_hmc_adapt \
      = compute_ess(samples_hmc_adapt, time_hmc_adapt)

    np.savetxt(os.path.join(path, 'ess_hmc_adapt.csv'), ess_hmc_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_hmc_adapt.csv'), [min_ess_per_sec_hmc_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_hmc_adapt.csv'), [min_ess_hmc_adapt], delimiter=",")
    pre_cond_operator, momentum_distribution = make_pre_cond_fn(pre_cond_params)

    if FLAGS.cov_type in ['log_space_independent_easier']:
      C_adapt_diag = pre_cond_operator.diag
      transformed_eigenvalues = np.sort((C_adapt_diag**2/(diags**2)).numpy())
    else:
      C_adapt = pre_cond_operator.to_dense().numpy()
      inv_M_adapt = np.matmul(C_adapt,C_adapt.transpose())

      #compute norm of transformed target
      transformed_hessian=tf.matmul(tf.matmul(C_adapt, tf.linalg.inv(target_dist.covariance()),
                                            adjoint_a=True), C_adapt)
      transformed_eigenvalues,_=tf.linalg.eigh(transformed_hessian)
      # save results
      if FLAGS.dims < 100:
        np.savetxt(os.path.join(path, 'inv_M_adapt.csv'), inv_M_adapt, delimiter = ",")
        sns.heatmap(inv_M_adapt)
        plt.savefig(os.path.join(path, 'inv_M_adapt.png'), bbox_inches = 'tight')
        plt.show()
        plt.close()

    np.savetxt(os.path.join(path, 'transformed_eigenvalues.csv'), transformed_eigenvalues, delimiter = ",")
    np.savetxt(os.path.join(path, 'is_accepted_hmc_adapt.csv'), np.mean(is_accepted_hmc_adapt,0), delimiter=",")

    if True:
      param = params_hmc_adapt.numpy().reshape([FLAGS.num_results, -1])
      plt.plot(param)
      plt.ylabel(' params')
      plt.savefig(os.path.join(path, 'param.png'), bbox_inches = 'tight')
      plt.show()
      plt.close()
      np.savetxt(os.path.join(path, 'param.csv'), param[-1], delimiter = ",")



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
    np.savetxt(os.path.join(path, 'time_hmc_learned.csv'), [time_hmc_learned], delimiter=",")

    #Compute ESS statistics
    ess_hmc_learned, mean_ess_hmc_learned, min_ess_hmc_learned, min_ess_per_sec_hmc_learned \
      = compute_ess(samples_hmc_learned, time_hmc_learned)

    is_accepted_hmc_learned, target_log_probs_hmc_learned = kernel_results_hmc_learned
    is_accepted_hmc_learned = tf.reduce_mean(tf.cast(is_accepted_hmc_learned,
      dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_hmc_learned.csv'), ess_hmc_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_hmc_learned.csv'), [min_ess_per_sec_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_hmc_learned.csv'), [min_ess_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_hmc_learned.csv'), [is_accepted_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'mean_ess_hmc_learned.csv'), [mean_ess_hmc_learned], delimiter=",")


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
      target_accept_prob = tf.cast(.7, dtype))

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

    ess_dual_hmc_adapt, _, min_ess_dual_hmc_adapt, \
    min_ess_per_sec_dual_hmc_adapt\
      = compute_ess(samples_dual_hmc_adapt, time_dual_hmc_adapt)

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
    ess_dual_hmc_learned, _, min_ess_dual_hmc_learned, min_ess_per_sec_dual_hmc_learned\
      = compute_ess(samples_dual_hmc_learned, time_dual_hmc_learned)

    is_accepted_dual_hmc_learned, target_log_probs_dual_hmc_learned = kernel_results_dual_hmc_learned
    is_accepted_dual_hmc_learned = tf.reduce_mean(tf.cast(is_accepted_dual_hmc_learned,
      dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_dual_hmc_learned.csv'), ess_dual_hmc_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_dual_hmc_learned.csv'), [min_ess_per_sec_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_dual_hmc_learned.csv'), [min_ess_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'time_dual_hmc_learned.csv'), [time_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_dual_hmc_learned.csv'), [is_accepted_dual_hmc_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'target_log_probs_dual_hmc_learned.csv'), target_log_probs_dual_hmc_learned, delimiter = ",")
    #np.save(os.path.join(path, 'samples_hmc_learned.npy'), samples_hmc_learned)

    plt.plot(target_log_probs_dual_hmc_learned)
    plt.ylabel('target_log_probs_dual_hmc_learned')
    plt.savefig(os.path.join(path, 'target_log_probs_dual_hmc_learned.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

    #ESS for target log prob observable
    ess_target_dual_hmc_learned, _, min_ess_target_dual_hmc_learned, \
    min_ess_target_per_sec_dual_hmc_learned \
        = compute_ess(target(samples_dual_hmc_learned), time_dual_hmc_learned)
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


  #elif FLAGS.adaptation == 'nuts':
  if FLAGS.num_leapfrog_steps == 1 :
    #####
    #Compare with NUTS
    #####


    nuts_sampler = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn = target,
      step_size = 0.1,
      max_tree_depth=5)

    adaptive_nuts_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel=nuts_sampler,
      num_adaptation_steps=FLAGS.num_results,
      target_accept_prob=.7)


    @tf.function(autograph=False)
    def run_adaptive_nuts_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results if FLAGS.cov_type == 'log_space_independent_easier' else FLAGS.num_results//10,
        num_burnin_steps=FLAGS.num_burnin_steps//3 if FLAGS.cov_type == 'log_space_independent_easier' else FLAGS.num_burnin_steps,
        current_state=tf.cast(init_positions, tf.float32),
        kernel=adaptive_nuts_sampler,
        num_steps_between_results=0,
        return_final_kernel_results=False,)

      return samples, kernel_results


    start_time_nuts_adapt = time.time()
    samples_nuts_adapt,  kernel_results_nuts_adapt  = run_adaptive_nuts_chain()
    time_nuts_adapt = time.time() - start_time_nuts_adapt
    np.savetxt(os.path.join(path, 'time_nuts_adapt.csv'), [time_nuts_adapt], delimiter = ",")


    ess_nuts_adapt, _, min_ess_nuts_adapt, \
    min_ess_per_sec_nuts_adapt\
      = compute_ess(samples_nuts_adapt, time_nuts_adapt)

    nuts_adapted_step_sizes = kernel_results_nuts_adapt.new_step_size
    is_accepted_nuts_adapt = tf.reduce_mean(tf.cast(kernel_results_nuts_adapt.inner_results.is_accepted,
                                         dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_nuts_adapt.csv'), ess_nuts_adapt, delimiter=",")
    np.savetxt(os.path.join(path, 'nuts_adapted_step_sizes.csv'), nuts_adapted_step_sizes, delimiter = ",")
    np.savetxt(os.path.join(path, 'is_accepted_nuts_adapt.csv'), [is_accepted_nuts_adapt], delimiter=",")
    np.savetxt(os.path.join(path, 'steps_nuts_adapt.csv'), kernel_results_nuts_adapt.inner_results.leapfrogs_taken,
               delimiter=",")


    learned_nuts_sampler = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn = target,
      max_tree_depth = 5,
      step_size = tf.nest.map_structure(lambda a: a[-1], nuts_adapted_step_sizes))


    @tf.function(autograph=False)
    def run_learned_nuts_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results if FLAGS.cov_type == 'log_space_independent_easier' else FLAGS.num_results//10,
        num_burnin_steps=0,
        current_state=tf.cast(samples_nuts_adapt[-1], tf.float32),
        kernel=learned_nuts_sampler,
        num_steps_between_results=0,
        return_final_kernel_results=False,)
      return samples, kernel_results



    start_time_nuts_learned = time.time()
    samples_nuts_learned, kernel_results_nuts_learned = run_learned_nuts_chain()
    time_nuts_learned = time.time() - start_time_nuts_learned

    ess_nuts_learned, mean_ess_nuts_learned, min_ess_nuts_learned, \
    min_ess_per_sec_nuts_learned \
      = compute_ess(samples_nuts_learned, time_nuts_learned)

    is_accepted_nuts_learned = tf.reduce_mean(tf.cast(kernel_results_nuts_learned.is_accepted,
                                         dtype = tf.float32))

    np.savetxt(os.path.join(path, 'ess_nuts_learned.csv'), ess_nuts_learned, delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_per_sec_nuts_learned.csv'), [min_ess_per_sec_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'min_ess_nuts_learned.csv'), [min_ess_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'time_nuts_learned.csv'), [time_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'is_accepted_nuts_learned.csv'), [is_accepted_nuts_learned], delimiter=",")
    np.savetxt(os.path.join(path, 'steps_nuts_learned.csv'), kernel_results_nuts_learned.leapfrogs_taken, delimiter=",")
    np.savetxt(os.path.join(path, 'mean_ess_nuts_learned.csv'), [mean_ess_nuts_learned], delimiter=",")


