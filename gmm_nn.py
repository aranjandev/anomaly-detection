import tensorflow as tf 
import numpy as np
import gmm_em
import logging
import logging.handlers
from matplotlib import pyplot as plt

def config_logging(loglevel):
    logger = logging.getLogger('lockcontrol')
    logger.setLevel(getattr(logging, loglevel.upper()))
    rfh = logging.handlers.RotatingFileHandler('localdata/sim.log', maxBytes=100000, backupCount=10)
    rfh.setFormatter(logging.Formatter(fmt='%(asctime)s[%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(rfh)
    logger.propagate = False
    return logger

LOGGER = config_logging(loglevel='INFO')

def get_loss(input, pred_gamma, lambda_1, lambda_2):
    """     
    pred_gamma: N x K
    input: N x D
    """
    N = pred_gamma.shape[0]
    gamma_k = tf.reduce_sum(pred_gamma, axis=0)
    gamma_k = tf.math.add(gamma_k, tf.constant(1e-12, dtype=tf.float64)) # add a small number to gamma to avoid divide by zero
    phi_k = gamma_k/N
    mu_k = tf.divide(tf.reduce_sum(tf.multiply(tf.expand_dims(pred_gamma, 2), tf.expand_dims(input, 1)), axis=0), tf.expand_dims(gamma_k, 1))
    x_mu = tf.subtract(tf.expand_dims(input, 1), tf.expand_dims(mu_k, 0))
    
    gamma_expanded = tf.expand_dims(tf.expand_dims(pred_gamma, 2), 3)
    
    sigma_k_unnorm = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(x_mu, 2), tf.expand_dims(x_mu, 3)), gamma_expanded), axis=0)    
    
    red_gamma = tf.reduce_sum(gamma_expanded, axis=0)
    
    sigma_k = tf.divide(sigma_k_unnorm, red_gamma)
    
    inv_sigma_k = tf.linalg.inv(sigma_k)
    x_mu_sig_x_mu = -0.5 * tf.squeeze((tf.expand_dims(x_mu, 2) @ inv_sigma_k) @ tf.expand_dims(x_mu, 3))
    exp_xmusigxmu = tf.math.exp(x_mu_sig_x_mu)
    sig_factor_k = tf.divide(phi_k, tf.math.sqrt(2 * tf.constant(np.pi, dtype=tf.float64) * tf.linalg.det(sigma_k)))
    E_zi = -1 * tf.math.log(tf.reduce_sum(tf.multiply(exp_xmusigxmu, sig_factor_k), axis=1))
    loss_term1 = tf.reduce_sum(E_zi, axis=0)
    diag_sum = tf.reduce_sum(tf.linalg.trace(1/sigma_k))
    loss_term2 = diag_sum
    loss = lambda_1 * loss_term1 / N + lambda_2 * loss_term2
    return loss, phi_k, mu_k, sigma_k

lambda_1 = 1e-1
lambda_2 = 1e-4
initializer = tf.keras.initializers.GlorotNormal()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, kernel_initializer=initializer, activation=tf.nn.relu, input_shape=(gmm_em.DIMENSIONS, ), dtype=tf.float64),
    tf.keras.layers.Dropout(0.5, dtype=tf.float64),
    tf.keras.layers.BatchNormalization(dtype=tf.float64),
    tf.keras.layers.Dense(64, kernel_initializer=initializer, activation=tf.nn.relu, dtype=tf.float64),
    tf.keras.layers.Dropout(0.5, dtype=tf.float64),
    tf.keras.layers.BatchNormalization(dtype=tf.float64),    
    tf.keras.layers.Dense(10, kernel_initializer=initializer, activation=tf.nn.softmax, dtype=tf.float64)
])

input = tf.Variable(gmm_em.generate_gmm_data(gmm_em.DATA_POINTS, gmm_em.CLUSTERS, gmm_em.DIMENSIONS)[0], dtype=tf.float64)
x = input
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005) 
all_loss = []
for i in range(1000):
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss, phi_k, mu_k, sigma_k = get_loss(x, prediction, lambda_1=lambda_1, lambda_2=lambda_2)
        gradients = tape.gradient(loss, model.trainable_variables)
        #LOGGER.info('gradients: {}'.format([tf.reduce_sum(grad).numpy() for grad in gradients]))
        #LOGGER.info('{}'.format(gradients))
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        all_loss += loss.numpy()
        LOGGER.info('iter {}: loss = {}'.format(i, loss))
        if i == 1:
            gmm_em.plot_fitted_data(x.numpy(), mu_k.numpy(), sigma_k.numpy())

gmm_em.plot_fitted_data(x.numpy(), mu_k.numpy(), sigma_k.numpy())
plt.plot(all_loss)