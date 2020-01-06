import tensorflow as tf 
import gmm_em

# N samples
# K clusters
# D features

def loss(pred_gamma, input):
    """     
    pred_gamma: N x K
    input: N x D
    """
input = tf.Variable(gmm_em.generate_gmm_data(gmm_em.DATA_POINTS, gmm_em.CLUSTERS, gmm_em.DIMENSIONS)[0], dtype=tf.float64)
pred_gamma = tf.Variable(np.tile([1/gmm_em.CLUSTERS]*gmm_em.CLUSTERS, [gmm_em.DATA_POINTS,1]), dtype=tf.float64)
N = pred_gamma.shape[0]
gamma_k = tf.reduce_sum(pred_gamma, axis=0)
phi_k = gamma_k/N
means = tf.divide(tf.reduce_sum(tf.multiply(tf.expand_dims(pred_gamma, 2), tf.expand_dims(input, 1)), axis=0), tf.expand_dims(gamma_k, 1))
x_mu = tf.subtract(tf.expand_dims(input, 1), tf.expand_dims(means, 0))
gamma_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma_k, 0), 2), 3)
sigma_k_unnorm = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(x_mu, 2), tf.expand_dims(x_mu, 3)), gamma_expanded), axis=0)
sigma_k = tf.divide(sigma_k_unnorm, tf.reduce_sum(gamma_expanded, axis=0))
inv_sigma_k = tf.linalg.inv(sigma_k)
x_mu_sig_x_mu = -0.5 * tf.squeeze((tf.expand_dims(x_mu, 2) @ inv_sigma_k) @ tf.expand_dims(x_mu, 3))
exp_xmusigxmu = tf.math.exp(x_mu_sig_x_mu)
sig_factor_k = tf.divide(phi_k, tf.math.sqrt(2 * tf.constant(np.pi, dtype=tf.float64) * tf.linalg.det(sigma_k)))
E_zi = -1 * tf.math.log(tf.reduce_sum(tf.multiply(exp_xmusigxmu, sig_factor_k), axis=1))

#loss = lambda_1 * tf.reduce_sum(E_zi, axis=0) +     

# def getmodel(in_shape, out_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(8, activation=tf.nn.relu, input_shape=in_shape),
#         tf.keras.layers.Dense(8, activation=tf.nn.relu),
#         tf.keras.layers.Dense(out_shape, activation=tf.nn.softmax)
#     ])
