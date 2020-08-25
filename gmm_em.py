import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import logging
import logging.handlers
import gmm_nn

DIMENSIONS = 2
CLUSTERS = 10
DATA_POINTS = 5000
TRAINING_STEPS = 1000
TOLERANCE = 1e-5

def config_logging(loglevel):
    logger = logging.getLogger('lockcontrol')
    logger.setLevel(getattr(logging, loglevel.upper()))
    rfh = logging.handlers.RotatingFileHandler('localdata/sim.log', maxBytes=100000, backupCount=10)
    rfh.setFormatter(logging.Formatter(fmt='%(asctime)s[%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(rfh)
    logger.propagate = False
    return logger

# calculate condition number of the passed matrix
def tf_cond(x):
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]
    x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
    r_nan = tf.math.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(np.inf, r.dtype))
    tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
    return r

def is_invertible(x, epsilon=1e-6):
    eps_inv = tf.cast(1/epsilon, x.dtype)
    x_cond = tf_cond(x)
    return tf.math.is_finite(x_cond) & (x_cond < eps_inv)

def generate_gmm_data(points, components, dimensions):
    """Generates synthetic data of a given size from a random GMM"""
    np.random.seed(10)

    c_means = np.random.normal(size=[components, dimensions]) * 10
    c_variances = np.abs(np.random.normal(size=[components, dimensions]))
    c_weights = np.abs(np.random.normal(size=[components]))
    c_weights /= np.sum(c_weights)

    result = np.zeros((points, dimensions), dtype=np.float32)

    for i in range(points):
        comp = np.random.choice(np.array(range(10)), p=c_weights)
        result[i] = np.random.multivariate_normal(
            c_means[comp], np.diag(c_variances[comp])
        )

    np.random.seed()

    return result, c_means, c_variances, c_weights

def plot_gaussian(mean, covariance, color, zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    plt.plot(mean[0], mean[1], color[0] + ".", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigvals) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(pat.Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color=color, linewidth=1, zorder=zorder
    ))

def plot_fitted_data(data, means, covariances, true_means=None, true_covariances=None):
    """Plots the data and given Gaussian components"""
    plt.plot(data[:, 0], data[:, 1], "b.", markersize=0.5, zorder=0)

    if true_means is not None:
        for i in range(len(true_means)):
            plot_gaussian(true_means[i], true_covariances[i], "green", 1)

    for i in range(len(means)):
        plot_gaussian(means[i], covariances[i], "red", 2)

    plt.show()


def init_w_kmeans(data):
    """Calculate initialization values using K-Means"""
    # initialize means
    km = KMeans(n_clusters=CLUSTERS)
    labs = km.fit_predict(data)
    means = km.cluster_centers_
    # initialize covariaces
    covs = np.empty((CLUSTERS, DIMENSIONS, DIMENSIONS))
    for l in np.unique(labs.ravel()):
        ce = EmpiricalCovariance()
        ce.fit(data[labs==l, :])
        ce.fit(data)
        covs[l,:,:] = ce.covariance_
    return means, covs

def init_random(data):
    """Randomly select means from data points and covariance as identity"""
    means = data[np.random.choice(DATA_POINTS, CLUSTERS), :]
    covs = np.tile(np.eye(DIMENSIONS).reshape(1, DIMENSIONS, DIMENSIONS), [CLUSTERS,1,1])
    return means, covs

def fit_gmm_em(input):
    """ Fit GMM to the input using EM algorithm. Init randomly. """
    bad_init = True
    while bad_init:
        print('Initing...')
        #means_init, covs_init = init_w_kmeans(input)    
        means_init, covs_init = init_random(input)
        alpha = tf.random.uniform([1,CLUSTERS])
        alpha = alpha / CLUSTERS
        means = tf.Variable(means_init)
        covs = tf.Variable(covs_init, dtype=tf.float32)

        plot_fitted_data(input, means.numpy(), covs.numpy())

        last_logLE = -np.inf
        for iter in range(TRAINING_STEPS):
            # e step    
            cond_num = tf_cond(covs)
            #print('cond# {}'.format(cond_num))
            if not tf.reduce_all(is_invertible(covs)):
                print('--- HIGH COND NUM')
            inv_covs = tf.linalg.inv(covs)    
            x_mu = tf.subtract(tf.expand_dims(input, 0), tf.expand_dims(means, 1))
            x_mu_sig_x_mu = -0.5 * tf.expand_dims(tf.reduce_sum(tf.multiply(x_mu @ inv_covs, x_mu), axis=2), 1)
            log_alpha = tf.reshape(tf.math.log(alpha), [CLUSTERS,1,1])
            ln2pi_const = -0.5 * DIMENSIONS * tf.math.log(2*tf.constant(np.pi))
            lnsigdet = tf.reshape(-0.5 * tf.math.log(tf.linalg.det(covs)), [CLUSTERS,1,1])
            log_rD = tf.squeeze(log_alpha + ln2pi_const + lnsigdet + x_mu_sig_x_mu)
            D = tf.reduce_sum(tf.math.exp(log_rD), axis=0)
            if tf.reduce_any(tf.math.equal(D, 0)):
                print('Bad initialization!')
                break
            else:
                bad_init = False
            log_LE = tf.reduce_sum(tf.math.log(D))/DATA_POINTS
            r = tf.math.exp(log_rD)/ tf.expand_dims(D,0)

            print('LE estimate ({}/{}): {}'.format(iter, TRAINING_STEPS, log_LE))
            if np.abs(log_LE.numpy() - last_logLE) < TOLERANCE:
                break
            last_logLE = log_LE.numpy()

            # m step
            n_soft = tf.reduce_sum(r, axis=1)
            alpha_new = n_soft/tf.reduce_sum(n_soft)
            means_new = (r @ input)/tf.expand_dims(n_soft, 1)
            x_mu_new = tf.subtract(tf.expand_dims(input, 0), tf.expand_dims(means_new, 1))
            covs_new = tf.reduce_sum(tf.reshape(r,[r.shape[0], r.shape[1], 1, 1]) * tf.expand_dims(x_mu_new, 3) @ tf.expand_dims(x_mu_new, 2), axis=1) / tf.reshape(n_soft, [n_soft.shape[0], 1, 1])

            if tf.math.reduce_any(tf.math.is_nan(covs_new)):
                print('ERROR: covs_new has nan')

            alpha = alpha_new
            means = means_new
            covs = covs_new

    plot_fitted_data(input, means.numpy(), covs.numpy())

def main():
    logger = config_logging(loglevel='INFO')
    for i in range(5):
        input,_,_,_ = generate_gmm_data(DATA_POINTS, CLUSTERS, DIMENSIONS)
        # print('----- Running EM --------')
        # fit_gmm_em(input)
        print('----- Running DNN --------')
        mu_k, sigma_k, phi_k = gmm_nn.fit_gmm_nn(input, logger)
        plot_fitted_data(input, mu_k, sigma_k)

if __name__ == "__main__":
    main()