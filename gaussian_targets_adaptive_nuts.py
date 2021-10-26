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
tfd = tfp.distributions
tfb = tfp.bijectors


flags.DEFINE_integer("num_chains",
                     default=10,
                     help="number of parallel MCMC chains")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=1,
                     help="number of leapfrog steps")
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
flags.DEFINE_string(
    'adaptation',
    default='nuts_windows',
    help="Adaptation type.")


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


  #elif FLAGS.adaptation == 'nuts':
  if FLAGS.num_leapfrog_steps == 1 :
    #####
    #Compare with NUTS
    #####
    windowed_adaptive_nuts = tfp.experimental.mcmc.windowed_adaptive_nuts(
      n_draws = 1,
      joint_dist = target_dist,
      n_chains = FLAGS.num_chains,
      num_adaptation_steps = FLAGS.num_results,
      current_state = tf.cast(init_positions, tf.float32),
      init_step_size = .1,
      max_tree_depth = 5,
      return_final_kernel_results = True,
      discard_tuning = False)


    np.savetxt(os.path.join(path, 'nuts_adapted_variance.csv'),
               windowed_adaptive_nuts.final_kernel_results.momentum_distribution.variance()[0],
               delimiter=",")


    learned_nuts_sampler = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
      step_size = windowed_adaptive_nuts.final_kernel_results.step_size,
      target_log_prob_fn = target,
      max_tree_depth = 5,
      momentum_distribution = windowed_adaptive_nuts.final_kernel_results.momentum_distribution
    )


    @tf.function(autograph=False)
    def run_learned_nuts_chain():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=FLAGS.num_results,
        num_burnin_steps=0,
        current_state=tf.cast(windowed_adaptive_nuts.all_states[-1], tf.float32),
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

