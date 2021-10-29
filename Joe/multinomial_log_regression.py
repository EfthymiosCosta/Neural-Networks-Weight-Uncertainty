import numpy as np
import matplotlib.pyplot as plt


def normal_sample(mean, cov, sample_size):
    sample = np.random.multivariate_normal(mean, cov, size=[sample_size])
    x_sample = [x[0] for x in sample]
    y_sample = [y[1] for y in sample]
    return x_sample, y_sample


def plot_points(points, color):
    x = [z[0] for z in points]
    y = [z[1] for z in points]
    plt.scatter(x, y, color=color)


def sigma_transform(rho):
    return np.log(np.ones(rho.shape) + np.exp(rho))


def log_prior(beta, tau):
    (k, m) = beta.shape
    return -np.linalg.norm(beta)**2/(2*tau*tau) - k * m * np.log(2*np.pi*tau*tau) / 2


def log_likelihood(y, x, beta):
    (n, m) = x.shape
    z = y @ beta @ np.transpose(x)
    return -np.sum(np.log(np.ones(n) + np.exp(z)))


def log_q(beta, mu, rho):
    sigma = sigma_transform(rho)
    z = (beta - mu)**2
    w = np.reciprocal(sigma**2)
    return - np.sum(np.log(2 * np.pi * sigma**2)) / 2 - np.sum(np.multiply(z, w)) / 2


def gradient(mu, rho, x, y, tau):
    (k, m) = mu.shape
    (n, _) = x.shape

    epsilon = np.random.normal(0, 1, size=(k, m))
    sigma = sigma_transform(rho)
    beta = mu + np.multiply(sigma, epsilon)

    z = y @ beta
    z = [np.dot(a, b) for a, b in zip(z, x)]
    z = np.ones(n) - np.reciprocal(1 + np.exp(z))
    z = np.transpose(np.repeat([z], repeats=m, axis=0))
    z = np.multiply(z, x)
    z = np.transpose(y) @ z

    grad_mu = z - beta / (tau**2)
    grad_rho = np.multiply(grad_mu, epsilon) + np.reciprocal(sigma)
    grad_rho = np.multiply(grad_rho, np.reciprocal(np.ones((k, m)) + np.exp(-rho)))

    return grad_mu, grad_rho


def calculate_ELBO(mu, rho, tau, x, y, samples):
    (k, _) = mu.shape
    (n, m) = x.shape
    sigma = sigma_transform(rho)
    total = 0

    for _ in range(samples):
        epsilon = np.random.normal(0, 1, size=(k, m))
        beta = mu + np.multiply(sigma, epsilon)

        # print(log_q(beta, mu, sigma))
        # print(log_prior(beta, tau=1))
        # print(log_likelihood(y, x, beta))

        total += - log_q(beta, mu, sigma) + log_prior(beta, tau) + log_likelihood(y, x, beta)

    return total / samples


def main():
    n, m, k = 100, 3, 4
    mean_1 = np.array([1, 3, 1])
    mean_2 = np.array([0, 0, 1])
    mean_3 = np.array([3, 1, 1])
    mean_4 = np.array([2, 5, 1])
    cov_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    cov_2 = cov_1
    cov_3 = cov_1
    cov_4 = cov_1

    x_1 = np.random.multivariate_normal(mean_1, cov_1, size=[n])
    x_2 = np.random.multivariate_normal(mean_2, cov_2, size=[n])
    x_3 = np.random.multivariate_normal(mean_3, cov_3, size=[n])
    x_4 = np.random.multivariate_normal(mean_4, cov_4, size=[n])
    y_1 = np.repeat([[1, 0, 0, 0]], axis=0, repeats=n)
    y_2 = np.repeat([[0, 1, 0, 0]], axis=0, repeats=n)
    y_3 = np.repeat([[0, 0, 1, 0]], axis=0, repeats=n)
    y_4 = np.repeat([[0, 0, 0, 1]], axis=0, repeats=n)

    x = np.concatenate((x_1, x_2, x_3, x_4))
    y = np.concatenate((y_1, y_2, y_3, y_4))

    # plot_points(x_1, 'red')
    # plot_points(x_2, 'blue')
    # plot_points(x_3, 'green')
    # plot_points(x_4, 'black')
    # plt.show()

    mu = np.zeros((k, m))
    rho = np.zeros((k, m))
    tau = 1

    alpha = 0.0001

    for iteration in range(10000):
        grad_mu, grad_rho = gradient(mu, rho, x, y, tau)
        mu -= alpha * grad_mu
        rho -= alpha * grad_rho
        # print(mu, rho)
        if iteration % 100 == 0:
            print("ELBO: %s" % calculate_ELBO(mu, rho, tau, x, y, 1000))


if __name__ == "__main__":
    main()
