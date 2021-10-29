import numpy as np
import matplotlib.pyplot as plt


def normal_sample(mean, cov, sample_size):
    sample = np.random.multivariate_normal(mean, cov, size=[sample_size])
    x_sample = [x[0] for x in sample]
    y_sample = [y[1] for y in sample]
    return x_sample, y_sample


def gradient(mu, rho, x, y, tau):
    (n, m) = x.shape
    epsilon = np.random.normal(0, 1, size=[m])
    sigma = sigma_transform(rho)
    beta = mu + np.multiply(sigma, epsilon)

    z = np.multiply(y, np.dot(x, beta))
    z = np.ones(n) + np.exp(z)
    z = np.ones(n) - np.reciprocal(z)
    z = np.multiply(y, z)

    grad_mu = - np.dot(np.transpose(x), z) + beta / (tau**2)

    # grad_rho = - np.multiply(epsilon**2, np.reciprocal(sigma)) - np.reciprocal(sigma) + \
    #              np.multiply(epsilon**2, np.reciprocal(sigma**3)) + np.multiply(epsilon, grad_mu)
    grad_rho = - np.reciprocal(sigma) + np.multiply(epsilon, grad_mu)
    grad_rho = np.multiply(grad_rho, np.reciprocal(np.ones(m) + np.exp(-rho)))

    return grad_mu, grad_rho


def plot_points(points, color):
    x = [z[0] for z in points]
    y = [z[1] for z in points]
    plt.scatter(x, y, color=color)


def show_selection(mu, rho, x_1, x_2, lines):
    plot_points(x_1, 'red')
    plot_points(x_2, 'blue')
    plt.xlim([-3, 6])
    plt.ylim([-2, 7])

    (_, m) = x_1.shape

    for _ in range(lines):
        epsilon = np.random.normal(0, 1, size=[m])
        sigma = sigma_transform(rho)
        beta = mu + np.multiply(sigma, epsilon)
        x = [-3, 6]
        y = [(- beta[2] - beta[0]*x_i) / beta[1] for x_i in x]
        plt.plot(x, y)

    plt.show()


def log_prior(beta, tau):
    (m,) = beta.shape
    return -np.linalg.norm(beta)**2/(2*tau*tau) - m * np.log(2*np.pi*tau*tau) / 2


def log_likelihood(y, x, beta):
    (n, m) = x.shape
    z = np.multiply(y, np.dot(x, beta))
    z = np.log(np.ones(n) + np.exp(z))
    return -sum(z)


def log_q(beta, mu, rho):
    sigma = sigma_transform(rho)
    z = (beta - mu)**2
    w = np.reciprocal(sigma**2)
    return - sum(np.log(2 * np.pi * sigma**2)) / 2 - np.dot(z, w) / 2


def calculate_ELBO(mu, rho, x, y, samples):
    (n, m) = x.shape
    sigma = sigma_transform(rho)
    total = 0

    for _ in range(samples):
        epsilon = np.random.normal(0, 1, size=[m])
        beta = mu + np.multiply(sigma, epsilon)

        # print(log_q(beta, mu, sigma))
        # print(log_prior(beta, tau=1))
        # print(log_likelihood(y, x, beta))

        total += - log_q(beta, mu, sigma) + log_prior(beta, tau=1) + log_likelihood(y, x, beta)

    return total / samples


def sigma_transform(rho):
    (m,) = rho.shape
    return np.log(np.ones(m) + np.exp(rho))


def main():
    n, m = 100, 3
    mean_1 = np.array([1, 3, 1])
    mean_2 = np.array([2, 1, 1])
    cov_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    cov_2 = cov_1

    # x_1, y_1 = normal_sample(mean_1, cov_1, n)
    # x_2, y_2 = normal_sample(mean_2, cov_2, n)

    # plt.scatter(x_1, y_1, color='red')
    # plt.scatter(x_2, y_2, color='blue')
    # plt.show()

    x_1 = np.random.multivariate_normal(mean_1, cov_1, size=[n])
    x_2 = np.random.multivariate_normal(mean_2, cov_2, size=[n])
    y_1 = np.ones(n)
    y_2 = - np.ones(n)
    x = np.concatenate((x_1, x_2))
    y = np.concatenate((y_1, y_2))

    mu = np.zeros(m)
    rho = np.zeros(m)
    sigma = sigma_transform(rho)

    epsilon = np.random.normal(0, 1, size=[m])
    beta = mu + np.multiply(sigma, epsilon)

    show_selection(mu, rho, x_1, x_2, 10)

    alpha = 0.0001

    for iteration in range(10000):
        grad_mu, grad_rho = gradient(mu, rho, x, y, tau=1)
        mu += alpha * grad_mu
        rho += alpha * grad_rho
        if iteration % 100 == 0:
            print(calculate_ELBO(mu, sigma, x, y, 10000))
            if iteration % 500 == 0:
                print(mu, rho)
                show_selection(mu, rho, x_1, x_2, 10)


if __name__ == "__main__":
    main()
