import tensorflow as tf 
import numpy as np
import gmm_em
from matplotlib import pyplot as plt

# input = tf.Variable(gmm_em.generate_gmm_data(gmm_em.DATA_POINTS, gmm_em.CLUSTERS, gmm_em.DIMENSIONS)[0], dtype=tf.float64)

def fit_gmm_nn(input, logger):
    def get_loss(input, pred_gamma, lambda_1, lambda_2):
        """     
        pred_gamma: N x K
        input: N x D
        """
        EPS = tf.constant(1e-10, dtype=tf.float64)
        N = pred_gamma.shape[0]
        gamma_k = tf.reduce_sum(pred_gamma, axis=0)
        gamma_k = tf.math.add(gamma_k, EPS) # add a small number to gamma to avoid divide by zero
        phi_k = gamma_k/N
        mu_k = tf.divide(tf.reduce_sum(tf.multiply(tf.expand_dims(pred_gamma, 2), tf.expand_dims(input, 1)), axis=0), tf.expand_dims(gamma_k, 1))
        x_mu = tf.subtract(tf.expand_dims(input, 1), tf.expand_dims(mu_k, 0))
        
        gamma_expanded = tf.expand_dims(tf.expand_dims(pred_gamma, 2), 3)
        
        sigma_k_unnorm = tf.reduce_sum(tf.multiply(tf.multiply(tf.expand_dims(x_mu, 2), tf.expand_dims(x_mu, 3)), gamma_expanded), axis=0)    
        
        red_gamma = tf.reduce_sum(gamma_expanded, axis=0)
        
        sigma_k = tf.divide(sigma_k_unnorm, red_gamma)
        sigma_k = tf.linalg.set_diag(sigma_k, tf.add(tf.linalg.diag_part(sigma_k), EPS))
        
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

    #define params
    lambda_2 = 1e-3
    lambda_1 = 1e-1
    N_TRAIN = 500
    learn_rate = 0.001

    # define model
    initializer = tf.keras.initializers.GlorotNormal()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, kernel_initializer=initializer, activation=tf.nn.relu, input_shape=(input.shape[1], ), dtype=tf.float64),
        tf.keras.layers.Dropout(0.3, dtype=tf.float64),
        tf.keras.layers.BatchNormalization(dtype=tf.float64),
        tf.keras.layers.Dense(64, kernel_initializer=initializer, activation=tf.nn.relu, dtype=tf.float64),
        tf.keras.layers.Dropout(0.3, dtype=tf.float64),
        tf.keras.layers.BatchNormalization(dtype=tf.float64),    
        tf.keras.layers.Dense(10, kernel_initializer=initializer, activation=tf.nn.softmax, dtype=tf.float64)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    # set up training
    x = tf.Variable(input, dtype=tf.float64)
    all_loss = []
    for i in range(N_TRAIN):
        with tf.GradientTape() as tape:
            prediction = model(x)
            loss, phi_k, mu_k, sigma_k = get_loss(x, prediction, lambda_1=lambda_1, lambda_2=lambda_2)
            gradients = tape.gradient(loss, model.trainable_variables)
            #logger.info('gradients: {}'.format([tf.reduce_sum(grad).numpy() for grad in gradients]))
            #logger.info('{}'.format(gradients))
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            all_loss += loss.numpy()
            logger.info('iter {}: loss = {}'.format(i, loss))
            # if i == 1:
            #     gmm_em.plot_fitted_data(x.numpy(), mu_k.numpy(), sigma_k.numpy())

    return mu_k.numpy(), sigma_k.numpy(), phi_k.numpy()    
