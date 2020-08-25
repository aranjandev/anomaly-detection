import tensorflow as tf 
import numpy as np
import gmm_em
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    lambda_2 = 0.0
    lambda_1 = 0.1
    N_TRAIN = 20
    learn_rate = 0.0005

    # define model
    #initializer = tf.keras.initializers.he_normal()
    #initializer = tf.keras.initializers.he_uniform()
    initializer = tf.keras.initializers.Constant(0.1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, 
                    kernel_initializer=initializer, 
                    #kernel_regularizer=tf.keras.regularizers.l2(0.01), 
                    #activation=tf.nn.relu, 
                    input_shape=(input.shape[1], ), 
                    dtype=tf.float64))
    model.add(tf.keras.layers.BatchNormalization(dtype=tf.float64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.2, dtype=tf.float64))
    model.add(tf.keras.layers.Dense(64, 
                    kernel_initializer=initializer, 
                    kernel_constraint=tf.keras.constraints.UnitNorm(),
                    #kernel_regularizer=tf.keras.regularizers.l2(0.01), 
                    #activation=tf.nn.relu, 
                    dtype=tf.float64))
    model.add(tf.keras.layers.BatchNormalization(dtype=tf.float64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5, dtype=tf.float64))    
    model.add(tf.keras.layers.Dense(10, kernel_initializer=initializer, activation=tf.nn.softmax, dtype=tf.float64))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    # set up training
    MINIBATCH_SZ = 32
    x = tf.Variable(input, dtype=tf.float64)
    all_loss = []    
    avg_grad_per_epoch = []
    for i in range(N_TRAIN):
        all_grad_means = []
        x = tf.random.shuffle(x)
        for batch_ind in tqdm(range(0, int(x.shape[0]/MINIBATCH_SZ))):
            batch = x[batch_ind*MINIBATCH_SZ:(batch_ind+1)*MINIBATCH_SZ, :]
            with tf.GradientTape() as tape:
                prediction = model(batch)
                loss, _, _, _ = get_loss(batch, prediction, lambda_1=lambda_1, lambda_2=lambda_2)
                gradients = tape.gradient(loss, model.trainable_variables)
                grad_means = [tf.reduce_mean(g) for g in gradients]
                all_grad_means.append(grad_means)
                #logger.info('gradients: {}'.format([tf.reduce_sum(grad).numpy() for grad in gradients]))
                #logger.info('{}'.format(gradients))
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                all_loss += loss.numpy()
                #logger.info('iter {}: loss = {}'.format(i, loss))
        full_prediction = model(x)
        full_loss, phi_k, mu_k, sigma_k = get_loss(x, full_prediction, lambda_1=lambda_1, lambda_2=lambda_2)
        print('iter {}: full training set loss = {}'.format(i, full_loss))
        grad_plot_data = np.asarray(all_grad_means)
        avg_grad_per_epoch.append([np.mean(all_grad_means, axis=0)])
    plt.plot(np.squeeze(np.asarray(avg_grad_per_epoch)))
    plt.legend([str(i) for i in range(10)])
    plt.grid(True)
    plt.show()
        # if i == 1:
        #     gmm_em.plot_fitted_data(x.numpy(), mu_k.numpy(), sigma_k.numpy())


    return mu_k.numpy(), sigma_k.numpy(), phi_k.numpy()    
