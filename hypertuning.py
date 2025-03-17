import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist

# fashion mnist
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)

class MLP:
    def __init__(self, inputs, hidden_size, outputs, hidden_layers, activation_function, weight_initialization="random", beta=0.9, beta1=0.9, beta2=0.999, beta_rmsprop=0.9, epsilon=1e-6):
        self.inputs = inputs
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.outputs = outputs
        self.activation_function = activation_function
        self.beta = beta
        self.beta_rmsprop = beta_rmsprop
        self.momentum_of_weights = []
        self.momentum_of_biases = []
        self.velocity_of_weights = []
        self.velocity_of_biases = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        weights = []

        if hidden_layers == 0:
            # case if there are no hidden layers
            if weight_initialization == "random":
                weights.append(np.random.uniform(
                    low=-2, high=2, size=(self.outputs, self.inputs)))
            elif weight_initialization == "Xavier":
                xavier_limit = np.sqrt(6 / (self.inputs + self.outputs))
                weights.append(np.random.uniform(
                    low=-xavier_limit, high=xavier_limit, size=(self.outputs, self.inputs)))
            self.momentum_of_weights.append(
                np.zeros((self.outputs, self.inputs)))
            self.momentum_of_biases.append(np.zeros((self.outputs, 1)))
            self.velocity_of_weights.append(
                np.zeros((self.outputs, self.inputs)))
            self.velocity_of_biases.append(np.zeros((self.outputs, 1)))
        else:
            if weight_initialization == "random":
                weights.append(np.random.uniform(
                    low=-2, high=2, size=(self.hidden_size, self.inputs)))
            elif weight_initialization == "Xavier":
                xavier_limit = np.sqrt(6 / (self.inputs + self.hidden_size))
                weights.append(np.random.uniform(
                    low=-xavier_limit, high=xavier_limit, size=(self.hidden_size, self.inputs)))
            self.momentum_of_weights.append(
                np.zeros((self.hidden_size, self.inputs)))
            self.momentum_of_biases.append(np.zeros((self.hidden_size, 1)))
            self.velocity_of_weights.append(
                np.zeros((self.hidden_size, self.inputs)))
            self.velocity_of_biases.append(np.zeros((self.hidden_size, 1)))

            for _ in range(0, hidden_layers - 1):
                if weight_initialization == "random":
                    weights.append(np.random.uniform(
                        low=-2, high=2, size=(self.hidden_size, self.hidden_size)))
                elif weight_initialization == "Xavier":
                    xavier_limit = np.sqrt(
                        6 / (self.hidden_size + self.hidden_size))
                    weights.append(np.random.uniform(
                        low=-xavier_limit, high=xavier_limit, size=(self.hidden_size, self.hidden_size)))
                self.momentum_of_weights.append(
                    np.zeros((self.hidden_size, self.hidden_size)))
                self.momentum_of_biases.append(np.zeros((self.hidden_size, 1)))
                self.velocity_of_weights.append(
                    np.zeros((self.hidden_size, self.hidden_size)))
                self.velocity_of_biases.append(np.zeros((self.hidden_size, 1)))

            if weight_initialization == "random":
                weights.append(np.random.uniform(
                    low=-2, high=2, size=(self.outputs, self.hidden_size)))
            elif weight_initialization == "Xavier":
                xavier_limit = np.sqrt(6 / (self.hidden_size + self.outputs))
                weights.append(np.random.uniform(
                    low=-xavier_limit, high=xavier_limit, size=(self.outputs, self.hidden_size)))
            self.momentum_of_weights.append(
                np.zeros((self.outputs, self.hidden_size)))
            self.momentum_of_biases.append(np.zeros((self.outputs, 1)))
            self.velocity_of_weights.append(
                np.zeros((self.outputs, self.hidden_size)))
            self.velocity_of_biases.append(np.zeros((self.outputs, 1)))

        self.weights = weights

        biases = []
        if hidden_layers == 0:
            # case if there are no hidden layers
            biases.append(np.random.uniform(
                low=-2, high=2, size=(self.outputs, 1)))
        else:
            biases.append(np.random.uniform(
                low=-2, high=2, size=(self.hidden_size, 1)))
            for _ in range(0, hidden_layers - 1):
                biases.append(np.random.uniform(
                    low=-2, high=2, size=(self.hidden_size, 1)))
            biases.append(np.random.uniform(
                low=-2, high=2, size=(self.outputs, 1)))
        self.biases = biases

    # convert vector to sigmoid
    def sigmoid(self, x):
        # print(x)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def activation(self, x, function):
        if function == "sigmoid":
            return self.sigmoid(x)
        elif function == "ReLU":
            return np.maximum(0, x)
        elif function == "tanh":
            return np.tanh(x)

    def derivative_of_activation(self, x, function):
        # function to find derivative of activation function
        if function == "sigmoid":
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        elif function == "ReLU":
            return np.where(x > 0, 1, 0)
        elif function == "tanh":
            return 1 - np.tanh(x)**2

    # convert vector to softmax
    def softmax(self, x):
        x_max = np.max(x, axis=0, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / e_x.sum(axis=0, keepdims=True)

    def feedforward(self, x_batch, weights_, biases_):
        # feedforward
        if x_batch[0].shape[0] != self.inputs:
            print(x_batch[0].shape)
            raise Exception("Input size does not match network input size")

        h_s, a_s = [], []
        a_s.append(np.dot(weights_[0], x_batch.T) + biases_[0])
        h_s.append(self.activation(a_s[0], function=self.activation_function))

        for i in range(1, self.hidden_layers):
            a_s.append(np.dot(weights_[i], h_s[-1]) + biases_[i])
            h_s.append(self.activation(
                a_s[-1], function=self.activation_function))
        a_s.append(np.dot(weights_[-1], h_s[-1]) + biases_[-1])
        h_s.append(self.softmax(a_s[-1]))
        return h_s, a_s

    def backpropogation(self, x, y, h_s, a_s, weights_):
        grad_a = [h_s[-1] - y.T]
        grad_h, grad_w, grad_b = [], [], []

        for k in range(self.hidden_layers, 0, -1):
            grad_w.append(np.dot(grad_a[-1], h_s[k-1].T))
            grad_b.append(grad_a[-1])
            grad_h.append(np.dot(weights_[k].T, grad_a[-1]))
            grad_a.append(
                grad_h[-1] * self.derivative_of_activation(a_s[k-1], function=self.activation_function))

        grad_w.append(np.dot(grad_a[-1], x))
        grad_b.append(grad_a[-1])
        grad_w.reverse()
        grad_b.reverse()
        return grad_w, grad_b

    def backpropogation_squared_loss(self, x, y, h_s, a_s, weights_):
        grad_a = [(h_s[-1] - y.T) * self.derivative_of_activation(a_s[-1],
                                                                  function=self.activation_function)]
        grad_h, grad_w, grad_b = [], [], []

        for k in range(self.hidden_layers, 0, -1):
            # EXTRA -1 IS BECAUSE OF 0 BASED INDEXING
            grad_w.append(np.dot(grad_a[-1], h_s[k-1].T))
            grad_b.append(grad_a[-1])
            grad_h.append(np.dot(weights_[k].T, grad_a[-1]))
            grad_a.append(
                grad_h[-1] * self.derivative_of_activation(a_s[k-1], function=self.activation_function))

        grad_w.append(np.dot(grad_a[-1], x))
        grad_b.append(grad_a[-1])
        grad_w.reverse()
        grad_b.reverse()
        return grad_w, grad_b

    def stochastic_gradient_descent(self, max_iter, x, y, x_validation, y_validation, eta, lambda_, loss="cross_entropy"):
        batch_size = 1
        for t in range(0, max_iter):
            print("Iteration", t, " of ", max_iter)
            for i in range(0, len(x), batch_size):
                x_batch, y_batch = x[i:i+batch_size], y[i:i+batch_size]
                h_s, a_s = self.feedforward(x_batch, self.weights, self.biases)
                grad_w, grad_b = self.backpropogation(
                    x_batch, y_batch, h_s, a_s, self.weights)
                for k in range(0, len(self.weights)):
                    # l2 regularization
                    grad_w[k] += lambda_ * self.weights[k]

                    self.weights[k] -= eta * grad_w[k]
                    self.biases[k] -= eta * grad_b[k]
            print("Calculating accuracy and loss")
            accuracy, loss = self.find_accuracy_and_cross_entropy_loss(x, y) if loss == "cross_entropy" else self.find_accuracy_and_squared_loss(
                x, y)
            print("Accuracy: ", accuracy)
            print("Calculating Validation accuracy and loss")
            val_accuracy, val_loss = self.find_accuracy_and_cross_entropy_loss(
                x_validation, y_validation) if loss == "cross_entropy" else self.find_accuracy_and_squared_loss(x_validation, y_validation)
            print("Validation Accuracy: ", val_accuracy)
            wandb.log({"accuracy": accuracy, "loss": loss,
                      "val_accuracy": val_accuracy, "val_loss": val_loss, "epoch": t+1})

        wandb.log({"accuracy": accuracy, "loss": loss,
                  "val_accuracy": val_accuracy, "val_loss": val_loss, "epoch": t+1})

    def momentum_update(self, eta, grad_w, grad_b):
        for j in range(len(self.weights)):
            self.momentum_of_weights[j] = self.beta *  self.momentum_of_weights[j] + eta * grad_w[j]
            self.momentum_of_biases[j] = self.beta *  self.momentum_of_biases[j] + eta * grad_b[j]

            self.weights[j] -= self.momentum_of_weights[j]
            self.biases[j] -= self.momentum_of_biases[j]

    def nesterov_update(self, eta, grad_w, grad_b):
        for j in range(len(self.weights)):
            lookahead_w = self.weights[j] -  self.beta * self.momentum_of_weights[j]
            lookahead_b = self.biases[j] -  self.beta * self.momentum_of_biases[j]

            self.momentum_of_weights[j] = self.beta *  self.momentum_of_weights[j] + eta * grad_w[j]
            self.momentum_of_biases[j] = self.beta *  self.momentum_of_biases[j] + eta * grad_b[j]

            self.weights[j] = lookahead_w - self.momentum_of_weights[j]
            self.biases[j] = lookahead_b - self.momentum_of_biases[j]

    def rmsprop_update(self, eta, grad_w, grad_b):
        for j in range(len(self.weights)):
            self.momentum_of_weights[j] = self.beta_rmsprop * self.momentum_of_weights[j] + (
                1 - self.beta_rmsprop) * (grad_w[j] ** 2)
            self.momentum_of_biases[j] = self.beta_rmsprop * self.momentum_of_biases[j] + (
                1 - self.beta_rmsprop) * (grad_b[j] ** 2)

            self.weights[j] -= (eta * grad_w[j]) / (np.sqrt(self.momentum_of_weights[j]) + self.epsilon)
            self.biases[j] -= (eta * grad_b[j]) / (np.sqrt(self.momentum_of_biases[j]) + self.epsilon)

    def adam_update(self, eta, grad_w, grad_b, update_count):
        for j in range(len(self.weights)):
            self.momentum_of_weights[j] = self.beta1 *  self.momentum_of_weights[j] + (1 - self.beta1) * grad_w[j]
            self.momentum_of_biases[j] = self.beta1 *  self.momentum_of_biases[j] + (1 - self.beta1) * grad_b[j]

            self.velocity_of_weights[j] = self.beta2 *  self.velocity_of_weights[j] + (1 - self.beta2) * (grad_w[j] ** 2)
            self.velocity_of_biases[j] = self.beta2 *  self.velocity_of_biases[j] + (1 - self.beta2) * (grad_b[j] ** 2)

            m_hat_w = self.momentum_of_weights[j] / (1 - self.beta1 ** update_count)
            m_hat_b = self.momentum_of_biases[j] / (1 - self.beta1 ** update_count)
            v_hat_w = self.velocity_of_weights[j] / (1 - self.beta2 ** update_count)
            v_hat_b = self.velocity_of_biases[j] / (1 - self.beta2 ** update_count)

            self.weights[j] -= eta * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[j] -= eta * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def gradient_descent(self, x, y, max_iter, optimizer="sgd", learning_rate=0.01, batch_size=64, lambda_=0, loss="cross_entropy"):
        print("Training for ", max_iter, " iterations",
              " with learning rate ", learning_rate, " and optimizer ", optimizer)
        t = 0
        eta = learning_rate

        # random validation split
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, test_size=0.2)
        x = x_train
        y = y_train

        batches = []
        for i in range(0, len(x), batch_size):
            batches.append((x[i:i+batch_size], y[i:i+batch_size]))

        if optimizer == "sgd":
            self.stochastic_gradient_descent(
                max_iter=max_iter, x=x, y=y, x_validation=x_validation, y_validation=y_validation, eta=eta, lambda_=lambda_, loss=loss)
            return

        update_count = 1
        while t < max_iter:
            t += 1
            print("Iteration", t, " of ", max_iter)
            for i in range(0, len(x), batch_size):
                x_batch, y_batch = x[i:i+batch_size], y[i:i+batch_size]
                grad_w, grad_b = [], []
                dw, db = [], []
                if optimizer == "nag":
                    look_ahead_w = []
                    for j in range(0, len(self.weights)):
                        look_ahead_w.append(
                            self.weights[j] - self.beta * self.momentum_of_weights[j])

                    look_ahead_b = []
                    for j in range(0, len(self.biases)):
                        look_ahead_b.append(
                            self.biases[j] - self.beta * self.momentum_of_biases[j])

                    h_s, a_s = self.feedforward(
                        x_batch, look_ahead_w, look_ahead_b)
                    grad_w, grad_b = self.backpropogation(
                        x_batch, y_batch, h_s, a_s, look_ahead_w) if loss == "cross_entropy" else self.backpropogation_squared_loss(
                        x_batch, y_batch, h_s, a_s, look_ahead_w)
                else:
                    h_s, a_s = self.feedforward(
                        x_batch, self.weights, self.biases)
                    grad_w, grad_b = self.backpropogation(
                        x_batch, y_batch, h_s, a_s, self.weights) if loss == "cross_entropy" else self.backpropogation_squared_loss(
                            x_batch, y_batch, h_s, a_s, self.weights)

                for j in range(0, len(self.weights)):
                    grad_w[j] += lambda_ * self.weights[j]
                    grad_b[j] = np.mean(grad_b[j], axis=1).reshape(-1, 1)

                if optimizer == "momentum":
                    # momentum based gradient descent rule
                    self.momentum_update(eta, grad_w, grad_b)
                elif optimizer == "nag":
                    # nestrov based gradient descent rule
                    self.nesterov_update(eta, grad_w, grad_b)
                elif optimizer == "rmsprop":
                    # rms prop gradient descent rule
                    self.rmsprop_update(eta, grad_w, grad_b)
                elif optimizer == "adam":
                    # adam gradient descent rule
                    self.adam_update(eta, grad_w, grad_b, update_count)
                elif optimizer == "nadam":
                    # nadam gradient descent rule
                    self.nadam_update(eta, grad_w, grad_b, update_count)
                update_count += 1

            print("Calculating accuracy and loss")
            accuracy, loss = self.find_accuracy_and_cross_entropy_loss(
                x, y) if loss == "cross_entropy" else self.find_accuracy_and_squared_loss(x, y)
            print("Accuracy: ", accuracy, "Loss: ", loss)

            print("Calculating Validation accuracy and loss")
            val_accuracy, val_loss = self.find_accuracy_and_cross_entropy_loss(
                x_validation, y_validation) if loss == "cross_entropy" else self.find_accuracy_and_squared_loss(x_validation, y_validation)
            print("Validation Accuracy: ", val_accuracy,
                  "Validation Loss: ", val_loss)

            wandb.log({"accuracy": accuracy, "loss": loss,
                      "val_accuracy": val_accuracy, "val_loss": val_loss, "epoch": t+1})

    def find_accuracy_and_squared_loss(self, x, y):
        correct, loss = 0, 0
        output, _ = self.feedforward(x, self.weights, self.biases)

        output = output[-1].T
        for i in range(len(x)):
            if np.argmax(output[i]) == np.argmax(y[i]):
                correct += 1
            loss += np.sum((output[i]-(y[i]))**2)

        return correct / len(x), loss / len(x)

    def find_accuracy_and_cross_entropy_loss(self, x, y):
        correct, loss = 0, 0
        output, _ = self.feedforward(x, self.weights, self.biases)

        output = output[-1].T
        output = np.clip(output, 1e-8, 1 - 1e-8)

        for i in range(len(x)):
            if np.argmax(output[i]) == np.argmax(y[i]):
                correct += 1
            loss -= np.sum(y[i] * np.log(output[i]))

        return correct / len(x), loss / len(x)

    def train(self, x, y, max_iter=10, learning_rate=0.01, optimizer="sgd", batch_size=32, lambda_=0, loss="cross_entropy"):
        self.gradient_descent(x, y, max_iter=max_iter, learning_rate=learning_rate,
                              optimizer=optimizer, batch_size=batch_size, lambda_=lambda_, loss=loss)

    def predict(self, x):
        h_s, a_s = self.feedforward(x, self.weights, self.biases)
        return h_s[-1]


def preprocessdata(x, y):
    x = x.reshape(x.shape[0], 784)
    x = x / 255

    one_hot_y = np.zeros((x.shape[0], np.max(y) + 1))

    one_hot_y[np.arange(x.shape[0]), y] = 1

    return x, one_hot_y

def initialize_model_train():
    wandb.init()
    config = wandb.config
    wandb.run.name=f"hl_{config.hidden_layers}_hn_{config.hidden_size}_bs_{config.batch_size}_ac_{config.activation_function}_op_{config.optimizer}"
    mlp = MLP(784, config.hidden_size, 10, config.hidden_layers, activation_function=config.activation_function, weight_initialization=config.weight_initialization, beta=0.9)
    x_preprocessed, y_preprocessed = preprocessdata(x_train, y_train)
    mlp.train(x_preprocessed, y_preprocessed, max_iter=config.max_iter, optimizer=config.optimizer, learning_rate=config.learning_rate, lambda_= config.weight_decay)
    wandb.finish()  

def main():
    sweep_config = {
        "name": "MLP Fashion MNIST Sweep",
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "hidden_size": {"values": [32, 64, 128]},
            "hidden_layers": {"values": [3, 4, 5]},
            "learning_rate": {"values": [0.001, 0.0001]},
            "max_iter": {"values": [5, 10]},
            "optimizer": {"values": ["momentum", "rmsprop", "nesterov", "adam", "sgd"]},
            "weight_initialization": {"values": ["random", "Xavier"]},
            "activation_function": {"values": ["sigmoid", "ReLU", "tanh"]},
            "batch_size": {"values": [32, 64, 128]},
            "weight_decay": {"values": [0, 0.0005, 0.5]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project="MLP")
    wandb.agent(sweep_id, function=initialize_model_train)

if __name__ == "__main__":
    main()
