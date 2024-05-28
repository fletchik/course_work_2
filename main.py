import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Real


def compute_trig_function_value(xdata, nstep=100):
    y = np.ones((xdata.shape[0]))
    for idx in range(nstep):
        y -= np.cos(np.pi * (idx + 1) * xdata.flatten()) / (idx + 1)
        y -= np.sin(np.pi * (idx + 1) * xdata.flatten()) / (idx + 1)
        y += np.cos(np.pi * (idx + 2) * xdata.flatten()) / (idx + 2)
        y += np.sin(np.pi * (idx + 2) * xdata.flatten()) / (idx + 2)
    return y


def compute_poly_function_value(xdata):
    y = 3 * xdata.flatten() ** 3 - 2 * xdata.flatten() ** 2 + xdata.flatten() + 5
    return y


def compute_complex_function_value(xdata):
    y = np.exp(xdata.flatten()) * (np.sin(2 * np.pi * xdata.flatten()) + np.cos(2 * np.pi * xdata.flatten()))
    return y


def compute_custom_function_value(xdata):
    y = (np.sin(xdata.flatten()) / 3) + (np.cos(2 * xdata.flatten()) * (xdata.flatten() / 5))
    return y


def compute_polynomial_features(xdata, npoly):
    ndata = xdata.shape[0]
    xpoly = np.zeros((ndata, npoly))
    xpoly[:, 0] = np.ones(ndata)
    for idx in range(1, npoly):
        xpoly[:, idx] = xdata.flatten() * xpoly[:, idx - 1]
    return xpoly


def compute_trigonometric_features(xdata, ntrig):
    ndata = xdata.shape[0]
    xtrig = np.zeros((ndata, 1 + 2 * ntrig))
    xtrig[:, 0] = np.ones((ndata))
    for idx in range(ntrig):
        xtrig[:, 2 * idx + 1] = np.cos(np.pi * (idx + 1.0) * xdata.flatten())
        xtrig[:, 2 * idx + 2] = np.sin(np.pi * (idx + 1.0) * xdata.flatten())
    return xtrig


def compute_loss_and_gradient(X, y, weights, alpha):
    predictions = X.dot(weights)
    errors = predictions - y
    loss = np.mean(errors ** 2) + alpha * np.sum(weights ** 2)
    gradient = 2 * X.T.dot(errors) / len(y) + 2 * alpha * weights
    return loss, gradient


def stochastic_gradient_descent(X, y, alpha=0.01, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
    np.random.seed(42)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * np.sqrt(1 / n_features)

    for iteration in range(n_iterations):
        random_index = np.random.randint(n_samples)
        xi = X[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]

        loss, gradient = compute_loss_and_gradient(xi, yi, weights, alpha)
        weights -= learning_rate * gradient

        if np.linalg.norm(gradient) < tolerance:
            break

    return weights


def compute_predictions(X, weights):
    return X.dot(weights)


def augment_data(xdata, frac):
    ndata = xdata.shape[0]
    noise = np.random.normal(loc=0, scale=0.1, size=(int(frac * ndata), 1))
    x_augmented = np.vstack((xdata, noise))
    return x_augmented


def train_model_sgd(xdata, ydata, model_type, npoly_or_ntrig, alpha=0.01, learning_rate=0.01, n_iterations=1000):
    if model_type == 'poly':
        X = compute_polynomial_features(xdata, npoly_or_ntrig)
    elif model_type == 'trig':
        X = compute_trigonometric_features(xdata, npoly_or_ntrig)
    else:
        raise ValueError("Invalid model type. Choose 'poly' or 'trig'.")

    weights = stochastic_gradient_descent(X, ydata, alpha, learning_rate, n_iterations)
    return weights


def evaluate_model(xdata, ydata, weights, model_type):
    if model_type == 'poly':
        X = compute_polynomial_features(xdata, len(weights))
    elif model_type == 'trig':
        X = compute_trigonometric_features(xdata, int((len(weights) - 1) / 2))
    else:
        raise ValueError("Invalid model type. Choose 'poly' or 'trig'.")

    ypred = compute_predictions(X, weights)
    mse = mean_squared_error(ydata, ypred)
    r2 = r2_score(ydata, ypred)
    return mse, r2, ypred


def train_and_evaluate_model(alpha, model_type, x_train, y_train, x_test, y_test, npoly_or_ntrig):
    weights = train_model_sgd(x_train, y_train, model_type, npoly_or_ntrig, alpha, learning_rate, n_iterations)
    mse, r2, ypred = evaluate_model(x_test, y_test, weights, model_type)
    return mse


def optimize_alpha(model_type, x_train, y_train, x_test, y_test, npoly_or_ntrig):
    search_space = [Real(1e-6, 1e1, "log-uniform", name='alpha')]
    result = gp_minimize(lambda alpha: train_and_evaluate_model(alpha[0], model_type, x_train, y_train, x_test, y_test, npoly_or_ntrig),
                          search_space, n_calls=50, random_state=0, n_initial_points=10)

    best_alpha = result.x[0]
    return best_alpha


def main():
    ndata = 3000

    # Generate data
    xdata = np.linspace(-1.0, 1.0, ndata).reshape(ndata, 1)

    # Augment data
    xdata_augmented = augment_data(xdata, frac=1.0)

    # Generate target values for polynomial, trigonometric, complex, and custom functions
    ydata_poly = compute_poly_function_value(xdata)
    ydata_trig = compute_trig_function_value(xdata)
    ydata_complex = compute_complex_function_value(xdata)
    ydata_custom = compute_custom_function_value(xdata)

    ydata_poly_augmented = compute_poly_function_value(xdata_augmented)
    ydata_trig_augmented = compute_trig_function_value(xdata_augmented)
    ydata_complex_augmented = compute_complex_function_value(xdata_augmented)
    ydata_custom_augmented = compute_custom_function_value(xdata_augmented)

    # Split data for models
    x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(xdata, ydata_poly, test_size=0.2, random_state=42)
    x_train_trig, x_test_trig, y_train_trig, y_test_trig = train_test_split(xdata, ydata_trig, test_size=0.2, random_state=42)
    x_train_complex, x_test_complex, y_train_complex, y_test_complex = train_test_split(xdata, ydata_complex, test_size=0.2, random_state=42)
    x_train_custom, x_test_custom, y_train_custom, y_test_custom = train_test_split(xdata, ydata_custom, test_size=0.2, random_state=42)

    x_train_poly_aug, x_test_poly_aug, y_train_poly_aug, y_test_poly_aug = train_test_split(xdata_augmented, ydata_poly_augmented, test_size=0.2, random_state=42)
    x_train_trig_aug, x_test_trig_aug, y_train_trig_aug, y_test_trig_aug = train_test_split(xdata_augmented, ydata_trig_augmented, test_size=0.2, random_state=42)
    x_train_complex_aug, x_test_complex_aug, y_train_complex_aug, y_test_complex_aug = train_test_split(xdata_augmented, ydata_complex_augmented, test_size=0.2, random_state=42)
    x_train_custom_aug, x_test_custom_aug, y_train_custom_aug, y_test_custom_aug = train_test_split(xdata_augmented, ydata_custom_augmented, test_size=0.2, random_state=42)

    # Parameters
    npoly = 10
    ntrig = 10
    global learning_rate, n_iterations
    learning_rate = 0.01
    n_iterations = 1000

    # Optimize alpha for each model
    best_alpha_poly = optimize_alpha('poly', x_train_poly, y_train_poly, x_test_poly, y_test_poly, npoly)
    best_alpha_trig = optimize_alpha('trig', x_train_trig, y_train_trig, x_test_trig, y_test_trig, ntrig)
    best_alpha_complex = optimize_alpha('trig', x_train_complex, y_train_complex, x_test_complex, y_test_complex, ntrig)
    best_alpha_custom = optimize_alpha('trig', x_train_custom, y_train_custom, x_test_custom, y_test_custom, ntrig)

    # Train models with optimized alpha
    weights_poly = train_model_sgd(x_train_poly, y_train_poly, 'poly', npoly, best_alpha_poly, learning_rate, n_iterations)
    weights_trig = train_model_sgd(x_train_trig, y_train_trig, 'trig', ntrig, best_alpha_trig, learning_rate, n_iterations)
    weights_complex = train_model_sgd(x_train_complex, y_train_complex, 'trig', ntrig, best_alpha_complex, learning_rate, n_iterations)
    weights_custom = train_model_sgd(x_train_custom, y_train_custom, 'trig', ntrig, best_alpha_custom, learning_rate, n_iterations)

    # Evaluate models on test data
    mse_poly, r2_poly, ypred_poly = evaluate_model(x_test_poly, y_test_poly, weights_poly, 'poly')
    mse_trig, r2_trig, ypred_trig = evaluate_model(x_test_trig, y_test_trig, weights_trig, 'trig')
    mse_complex, r2_complex, ypred_complex = evaluate_model(x_test_complex, y_test_complex, weights_complex, 'trig')
    mse_custom, r2_custom, ypred_custom = evaluate_model(x_test_custom, y_test_custom, weights_custom, 'trig')

    # Evaluate models on training data
    mse_poly_train, r2_poly_train, _ = evaluate_model(x_train_poly, y_train_poly, weights_poly, 'poly')
    mse_trig_train, r2_trig_train, _ = evaluate_model(x_train_trig, y_train_trig, weights_trig, 'trig')
    mse_complex_train, r2_complex_train, _ = evaluate_model(x_train_complex, y_train_complex, weights_complex, 'trig')
    mse_custom_train, r2_custom_train, _ = evaluate_model(x_train_custom, y_train_custom, weights_custom, 'trig')

    # Train models with augmented data
    weights_poly_aug = train_model_sgd(x_train_poly_aug, y_train_poly_aug, 'poly', npoly, best_alpha_poly, learning_rate, n_iterations)
    weights_trig_aug = train_model_sgd(x_train_trig_aug, y_train_trig_aug, 'trig', ntrig, best_alpha_trig, learning_rate, n_iterations)
    weights_complex_aug = train_model_sgd(x_train_complex_aug, y_train_complex_aug, 'trig', ntrig, best_alpha_complex, learning_rate, n_iterations)
    weights_custom_aug = train_model_sgd(x_train_custom_aug, y_train_custom_aug, 'trig', ntrig, best_alpha_custom, learning_rate, n_iterations)

    # Evaluate augmented models on test data
    mse_poly_aug, r2_poly_aug, ypred_poly_aug = evaluate_model(x_test_poly_aug, y_test_poly_aug, weights_poly_aug, 'poly')
    mse_trig_aug, r2_trig_aug, ypred_trig_aug = evaluate_model(x_test_trig_aug, y_test_trig_aug, weights_trig_aug, 'trig')
    mse_complex_aug, r2_complex_aug, ypred_complex_aug = evaluate_model(x_test_complex_aug, y_test_complex_aug, weights_complex_aug, 'trig')
    mse_custom_aug, r2_custom_aug, ypred_custom_aug = evaluate_model(x_test_custom_aug, y_test_custom_aug, weights_custom_aug, 'trig')

    # Evaluate augmented models on training data
    mse_poly_train_aug, r2_poly_train_aug, _ = evaluate_model(x_train_poly_aug, y_train_poly_aug, weights_poly_aug, 'poly')
    mse_trig_train_aug, r2_trig_train_aug, _ = evaluate_model(x_train_trig_aug, y_train_trig_aug, weights_trig_aug, 'trig')
    mse_complex_train_aug, r2_complex_train_aug, _ = evaluate_model(x_train_complex_aug, y_train_complex_aug, weights_complex_aug, 'trig')
    mse_custom_train_aug, r2_custom_train_aug, _ = evaluate_model(x_train_custom_aug, y_train_custom_aug, weights_custom_aug, 'trig')

    # Plot results
    plt.figure(figsize=(14, 10))

    # Polynomial model
    plt.subplot(2, 2, 1)
    plt.scatter(x_test_poly, y_test_poly, color='blue', label='Test Data')
    plt.scatter(x_test_poly, ypred_poly, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Polynomial Model\nTest MSE: {mse_poly:.4f}, R2: {r2_poly:.4f}')
    plt.legend()

    # Trigonometric model
    plt.subplot(2, 2, 2)
    plt.scatter(x_test_trig, y_test_trig, color='blue', label='Test Data')
    plt.scatter(x_test_trig, ypred_trig, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Trigonometric Model\nTest MSE: {mse_trig:.4f}, R2: {r2_trig:.4f}')
    plt.legend()

    # Complex model
    plt.subplot(2, 2, 3)
    plt.scatter(x_test_complex, y_test_complex, color='blue', label='Test Data')
    plt.scatter(x_test_complex, ypred_complex, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Complex Model\nTest MSE: {mse_complex:.4f}, R2: {r2_complex:.4f}')
    plt.legend()

    # Custom model
    plt.subplot(2, 2, 4)
    plt.scatter(x_test_custom, y_test_custom, color='blue', label='Test Data')
    plt.scatter(x_test_custom, ypred_custom, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Custom Model\nTest MSE: {mse_custom:.4f}, R2: {r2_custom:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 10))

    # Augmented Polynomial model
    plt.subplot(2, 2, 1)
    plt.scatter(x_test_poly_aug, y_test_poly_aug, color='blue', label='Test Data')
    plt.scatter(x_test_poly_aug, ypred_poly_aug, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Augmented Polynomial Model\nTest MSE: {mse_poly_aug:.4f}, R2: {r2_poly_aug:.4f}')
    plt.legend()

    # Augmented Trigonometric model
    plt.subplot(2, 2, 2)
    plt.scatter(x_test_trig_aug, y_test_trig_aug, color='blue', label='Test Data')
    plt.scatter(x_test_trig_aug, ypred_trig_aug, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Augmented Trigonometric Model\nTest MSE: {mse_trig_aug:.4f}, R2: {r2_trig_aug:.4f}')
    plt.legend()

    # Augmented Complex model
    plt.subplot(2, 2, 3)
    plt.scatter(x_test_complex_aug, y_test_complex_aug, color='blue', label='Test Data')
    plt.scatter(x_test_complex_aug, ypred_complex_aug, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Augmented Complex Model\nTest MSE: {mse_complex_aug:.4f}, R2: {r2_complex_aug:.4f}')
    plt.legend()

    # Augmented Custom model
    plt.subplot(2, 2, 4)
    plt.scatter(x_test_custom_aug, y_test_custom_aug, color='blue', label='Test Data')
    plt.scatter(x_test_custom_aug, ypred_custom_aug, color='red', alpha=0.5, label='Model Prediction')
    plt.title(f'Augmented Custom Model\nTest MSE: {mse_custom_aug:.4f}, R2: {r2_custom_aug:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print MSE and R2 scores
    print("\nModel Evaluation (without augmentation):")
    print(f"Polynomial Model: Train MSE = {mse_poly_train:.4f}, Test MSE = {mse_poly:.4f}, Train R2 = {r2_poly_train:.4f}, Test R2 = {r2_poly:.4f}")
    print(f"Trigonometric Model: Train MSE = {mse_trig_train:.4f}, Test MSE = {mse_trig:.4f}, Train R2 = {r2_trig_train:.4f}, Test R2 = {r2_trig:.4f}")
    print(f"Complex Model: Train MSE = {mse_complex_train:.4f}, Test MSE = {mse_complex:.4f}, Train R2 = {r2_complex_train:.4f}, Test R2 = {r2_complex:.4f}")
    print(f"Custom Model: Train MSE = {mse_custom_train:.4f}, Test MSE = {mse_custom:.4f}, Train R2 = {r2_custom_train:.4f}, Test R2 = {r2_custom:.4f}")

    print("\nModel Evaluation (with augmentation):")
    print(f"Polynomial Model: Train MSE = {mse_poly_train_aug:.4f}, Test MSE = {mse_poly_aug:.4f}, Train R2 = {r2_poly_train_aug:.4f}, Test R2 = {r2_poly_aug:.4f}")
    print(f"Trigonometric Model: Train MSE = {mse_trig_train_aug:.4f}, Test MSE = {mse_trig_aug:.4f}, Train R2 = {r2_trig_train_aug:.4f}, Test R2 = {r2_trig_aug:.4f}")
    print(f"Complex Model: Train MSE = {mse_complex_train_aug:.4f}, Test MSE = {mse_complex_aug:.4f}, Train R2 = {r2_complex_train_aug:.4f}, Test R2 = {r2_complex_aug:.4f}")
    print(f"Custom Model: Train MSE = {mse_custom_train_aug:.4f}, Test MSE = {mse_custom_aug:.4f}, Train R2 = {r2_custom_train_aug:.4f}, Test R2 = {r2_custom_aug:.4f}")

# Call main function to execute the modeling and evaluation process
main()
