import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Генерация данных для сложной функции с шумом на увеличенном интервале
def generate_noisy_complex_data(n_points, noise_level, interval):
    x = np.linspace(0, interval, n_points)
    y = np.exp(x) * (np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x)) + np.random.normal(0, noise_level, n_points)
    return x, y

# Генерация данных для пользовательской функции с шумом на увеличенном интервале
def generate_noisy_custom_data(n_points, noise_level, interval):
    x = np.linspace(0, interval, n_points)
    y = (np.sin(x) / 3) + (np.cos(2 * x) * (x / 5)) + np.random.normal(0, noise_level, n_points)
    return x, y

# Улучшение функции вычисления тригонометрических признаков
def compute_trigonometric_features(xdata, ntrig):
    ndata = xdata.shape[0]
    xtrig = np.zeros((ndata, 1 + 4 * ntrig))
    xtrig[:, 0] = np.ones((ndata))
    for idx in range(ntrig):
        xtrig[:, 4 * idx + 1] = np.cos(np.pi * (idx + 1.0) * xdata.flatten())
        xtrig[:, 4 * idx + 2] = np.sin(np.pi * (idx + 1.0) * xdata.flatten())
        xtrig[:, 4 * idx + 3] = np.cos(2 * np.pi * (idx + 1.0) * xdata.flatten())
        xtrig[:, 4 * idx + 4] = np.sin(2 * np.pi * (idx + 1.0) * xdata.flatten())
    return xtrig

# Обучение модели с использованием стохастического градиентного спуска
def train_model_sgd_no_reg(X, y, learning_rate=0.001, n_iterations=10000, tolerance=1e-6):
    np.random.seed(42)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * np.sqrt(1 / n_features)

    for iteration in range(n_iterations):
        random_index = np.random.randint(n_samples)
        xi = X[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]

        predictions = xi.dot(weights)
        errors = predictions - yi
        gradient = 2 * xi.T.dot(errors) / len(yi)
        weights -= learning_rate * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

    return weights

# Обучение модели с регуляризацией Тихонова
def train_model_sgd_tikhonov(X, y, alpha=0.1, learning_rate=0.001, n_iterations=10000, tolerance=1e-6):
    np.random.seed(42)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * np.sqrt(1 / n_features)

    for iteration in range(n_iterations):
        random_index = np.random.randint(n_samples)
        xi = X[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]

        predictions = xi.dot(weights)
        errors = predictions - yi
        gradient = 2 * xi.T.dot(errors) / len(yi) + 2 * alpha * weights
        weights -= learning_rate * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

    return weights

# Оценка модели
def evaluate_model(X, y, weights):
    ypred = X.dot(weights)
    mse = mean_squared_error(y, ypred)
    r2 = r2_score(y, ypred)
    return mse, r2, ypred

def main():
    # Параметры для генерации данных
    n_points = 10000  # Увеличим количество точек данных
    noise_level = 0.005  # Уменьшим уровень шума
    interval = 2  # Ещё больше увеличим интервал

    # Генерация данных для complex функции
    x_complex, y_complex = generate_noisy_complex_data(n_points, noise_level, interval)
    x_complex = x_complex.reshape(-1, 1)

    # Генерация данных для custom функции
    x_custom, y_custom = generate_noisy_custom_data(n_points, noise_level, interval)
    x_custom = x_custom.reshape(-1, 1)

    # Разделение данных на обучающую и тестовую выборки
    x_train_complex, x_test_complex, y_train_complex, y_test_complex = train_test_split(x_complex, y_complex, test_size=0.2, random_state=42)
    x_train_custom, x_test_custom, y_train_custom, y_test_custom = train_test_split(x_custom, y_custom, test_size=0.2, random_state=42)

    # Вычисление тригонометрических признаков
    ntrig = 300  # Увеличим количество тригонометрических признаков
    X_train_complex = compute_trigonometric_features(x_train_complex, ntrig)
    X_test_complex = compute_trigonometric_features(x_test_complex, ntrig)
    X_train_custom = compute_trigonometric_features(x_train_custom, ntrig)
    X_test_custom = compute_trigonometric_features(x_test_custom, ntrig)

    # Обучение модели без регуляризации для complex функции
    learning_rate = 0.001  # Уменьшим скорость обучения
    n_iterations = 10000  # Увеличим количество итераций
    weights_no_reg_complex = train_model_sgd_no_reg(X_train_complex, y_train_complex, learning_rate, n_iterations)

    # Обучение модели с регуляризацией Тихонова для complex функции
    alpha = 0.1  # Параметр регуляризации
    weights_tikhonov_complex = train_model_sgd_tikhonov(X_train_complex, y_train_complex, alpha, learning_rate, n_iterations)

    # Оценка моделей на тестовых данных для complex функции
    mse_no_reg_complex, r2_no_reg_complex, ypred_no_reg_complex = evaluate_model(X_test_complex, y_test_complex, weights_no_reg_complex)
    mse_tikhonov_complex, r2_tikhonov_complex, ypred_tikhonov_complex = evaluate_model(X_test_complex, y_test_complex, weights_tikhonov_complex)

    # Обучение модели без регуляризации для custom функции
    weights_no_reg_custom = train_model_sgd_no_reg(X_train_custom, y_train_custom, learning_rate, n_iterations)

    # Обучение модели с регуляризацией Тихонова для custom функции
    weights_tikhonov_custom = train_model_sgd_tikhonov(X_train_custom, y_train_custom, alpha, learning_rate, n_iterations)

    # Оценка моделей на тестовых данных для custom функции
    mse_no_reg_custom, r2_no_reg_custom, ypred_no_reg_custom = evaluate_model(X_test_custom, y_test_custom, weights_no_reg_custom)
    mse_tikhonov_custom, r2_tikhonov_custom, ypred_tikhonov_custom = evaluate_model(X_test_custom, y_test_custom,
                                                                                    weights_tikhonov_custom)

    # Визуализация результатов для complex функции
    plt.figure(figsize=(14, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(x_test_complex, y_test_complex, color='blue', label='Тестовые данные')
    plt.scatter(x_test_complex, ypred_no_reg_complex, color='red', alpha=0.5,
                label='Прогноз модели (без регуляризации)')
    plt.title(f'Исследуемая модель с Аугментацией\nMSE: {mse_no_reg_complex:.4f}, R2: {r2_no_reg_complex:.4f}')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(x_test_complex, y_test_complex, color='blue', label='Тестовые данные')
    plt.scatter(x_test_complex, ypred_tikhonov_complex, color='green', alpha=0.5,
                label='Прогноз модели (с Тихоновской регуляризацией)')
    plt.title(
        f'Модель с Тихоновской регуляризацией\nMSE: {mse_tikhonov_complex:.4f}, R2: {r2_tikhonov_complex:.4f}')
    plt.legend()

    # Визуализация результатов для custom функции
    plt.subplot(2, 2, 3)
    plt.scatter(x_test_custom, y_test_custom, color='blue', label='Тестовые данные')
    plt.scatter(x_test_custom, ypred_no_reg_custom, color='red', alpha=0.5, label='Прогноз модели (без регуляризации)')
    plt.title(f'Исследуемая модель с Аугментацией\nMSE: {mse_no_reg_custom:.4f}, R2: {r2_no_reg_custom:.4f}')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(x_test_custom, y_test_custom, color='blue', label='Тестовые данные')
    plt.scatter(x_test_custom, ypred_tikhonov_custom, color='green', alpha=0.5,
                label='Прогноз модели (с Тихоновской регуляризацией)')
    plt.title(
        f'Модель с Тихоновской регуляризацией\nMSE: {mse_tikhonov_custom:.4f}, R2: {r2_tikhonov_custom:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Печать MSE и R2
    print(f"Complex функция без регуляризации:\nMSE = {mse_no_reg_complex:.4f}, R2 = {r2_no_reg_complex:.4f}")
    print(
        f"Complex функция с Тихоновской регуляризацией:\nMSE = {mse_tikhonov_complex:.4f}, R2 = {r2_tikhonov_complex:.4f}")
    print(f"Custom функция без регуляризации:\nMSE = {mse_no_reg_custom:.4f}, R2 = {r2_no_reg_custom:.4f}")
    print(
        f"Custom функция с Тихоновской регуляризацией:\nMSE = {mse_tikhonov_custom:.4f}, R2 = {r2_tikhonov_custom:.4f}")


# Вызов основной функции для выполнения процесса моделирования и оценки
main()

