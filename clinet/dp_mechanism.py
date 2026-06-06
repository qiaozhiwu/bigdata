import numpy as np
from scipy.optimize import fsolve, minimize


def cal_client_sensitivity(lr, clip, dataset_size):
    """
    计算本地客户端学习的敏感度
    :param lr: 客户端学习的学习率
    :param clip: 梯度裁剪范数
    :param dataset_size: 本地数据集大小
    :return sensitivity:函数敏感度
    """
    sensitivity = 2 * lr * clip / dataset_size
    # print("sensitivity", sensitivity)
    return sensitivity


def laplace_mechanism(epsilon, sensitivity, size):
    """
    生成使大小为size，敏感度为sensitivity的数据满足ε-差分隐私的噪声
    :param epsilon: 隐私预算ε
    :param sensitivity: 函数敏感度
    :param size: 数据大小，即生成的噪声列表大小
    :return :生成的噪声列表
    """
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)


def gaussian_mechanism(epsilon, delta, sensitivity, size):
    """
    生成使大小为size，敏感度为sensitivity的数据满足(ε,δ)-差分隐私的噪声
    :param epsilon: 隐私预算ε
    :param delta: (ε,δ)-差分隐私的参数δ
    :param sensitivity: 函数敏感度
    :param size: 数据大小，即生成的噪声列表大小
    :return :生成的噪声列表
    """
    if epsilon >= 1:
        raise ValueError("Epsilon of Gaussian mechanism should be greater than 0 and less than 1!")
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)


def simple_composition(k, total_epsilon, total_delta):
    """
    根据总体隐私预算使用简单组合定理计算每步的差分隐私预算
    sources:Boosting and Differential Privacy
    :param k:参与轮数
    :param total_epsilon:总体隐私预算
    :param total_delta:总体隐私预算
    :return:每轮的隐私预算ε,δ
    """
    epsilon = total_epsilon / k
    delta = total_delta / k
    return epsilon, delta


def advanced_composition(k, total_epsilon, total_delta):
    """
    根据总体隐私预算使用高级组合定理计算每步的差分隐私预算
    sources:Boosting and Differential Privacy
    :param k:参与轮数
    :param total_epsilon:总体隐私预算
    :param total_delta:总体隐私预算
    :return:每轮的隐私预算ε,δ
    """
    delta_prime_ratio = 0.5
    delta_prime = delta_prime_ratio * total_delta
    delta = (total_delta - delta_prime) / k
    x0 = [total_epsilon / k]

    def f(x0):
        x = x0[0]
        return [np.sqrt(2 * k * np.log(1 / delta_prime)) * x + k * x * (np.exp(x) - 1) - total_epsilon]

    epsilon = fsolve(f, x0)
    result = epsilon, delta
    return result


def renyi_gaussian_composition(k, total_epsilon, total_delta):
    """
    根据总体隐私预算计算每步的隐私预算
    sources:Rényi differential privacy
    :param k:参与轮数
    :param total_epsilon:总体隐私预算
    :param total_delta:总体隐私预算
    :return:
    """
    n = 1 + np.log(1 / total_delta) / total_epsilon
    alpha = n + np.sqrt(n ** 2 - n)
    epsilon = (total_epsilon - np.log(1 / total_delta) / (alpha - 1)) / k
    return alpha, epsilon


def renyi_gaussian_mechanism(alpha, epsilon, sensitivity, size):
    """
    生成使大小为size，敏感度为sensitivity的数据满足(α,ε)-Renyi差分隐私的噪声
    :param alpha: α
    :param epsilon: (α,ε)-差分隐私的参数ε
    :param sensitivity: 函数敏感度
    :param size: 数据大小，即生成的噪声列表大小
    :return :生成的噪声列表
    """
    noise_scale = np.sqrt(alpha / (2 * epsilon)) * sensitivity
    # print("noise_scale:", noise_scale)
    return np.random.normal(0, noise_scale, size=size)


# todo
def Gaussian_moment(epsilon, delta, sensitivity, size):
    return


if __name__ == '__main__':
    gaussian_mechanism(1.1, 1e-7, 20, 100)
