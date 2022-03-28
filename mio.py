from textwrap import indent
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MSE(x, y):
    return np.sum((x - y)**2) / len(x)

class Activation:
    def __init__(self):
        self.f = None
        self.df = None

class SigmoidActivation(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def grad_sigmoid(x):
            return sigmoid(x) * (1 - sigmoid(x))
        self.f = sigmoid
        self.df = grad_sigmoid

class LinearActivation(Activation):
    def __init__(self):
        self.f = lambda x: x
        self.df = lambda x: 1

class ActivationStrategy:
    def __init__(self):
        pass

    def execute(self, architecture):
        raise NotImplementedError

class SigmoidLinearActivationStrategy:
    def __init__(self):
        self.sigmoid = SigmoidActivation()
        self.linear = LinearActivation()

    def execute(self, architecture: (int)):
        length = len(architecture)
        return [self.sigmoid] * (length - 2) + [self.linear]
        
class Initializer:
    def __init__(self):
        pass
    def get(self, shape):
        raise NotImplementedError
    def __call__(self, shape):
        return self.get(shape)

class ZeroInitializer(Initializer):
    def get(self, shape):
        return np.zeros(shape)

class UniformInitializer(Initializer):
    def __init__(self, lower=-1, higher=1):
        self.lower = lower
        self.higher = higher

    def get(self, shape):
        return np.random.random(shape) * (self.higher - self.lower) + self.lower

class LayerFactory:
    def __init__(self, initializer: Initializer):
        self.initializer = initializer

    def get(self, layer_shape, activation):
        inbound_shape, neurons = layer_shape
        return Layer(
            weights = self.initializer((inbound_shape, neurons)),
            bias = self.initializer((1, neurons)),
            activation = activation
        )

class Layer:   
    def __init__(self, weights: np.array, bias: np.array, activation: Activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def feedforward(self, inputs):
        intensities = inputs.dot(self.weights)
        inbound = intensities + self.bias
        outbound = self.activation.f(inbound)
        return inbound, outbound
    
    def apply(self, inputs):
        intensities = inputs @ self.weights
        return self.activation.f(intensities + self.bias)
    
    def __str__(self):
        from textwrap import indent 
        return "Layer(\n{}\n)".format(
            indent(f"Weights:\n{repr(self.weights)}\nBias:\n{repr(self.bias)}", "  ")
        )

class NN:
    def __init__(self, layers):
        self.layers = layers
        
    def apply(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def feedforward(self, inputs):
        x = inputs
        self.last_input = inputs
        inbound_list = []
        outbound_list = []
        for layer in self.layers:
            inbound, x = layer.feedforward(x)
            inbound_list.append(inbound)
            outbound_list.append(x)
        return inbound_list, outbound_list
    
    def calculate_errors(self, y, inbound, outbound):
        errors = [None] * len(self.layers)
        yhat = outbound[-1]
        
        errors[-1] = (yhat - y)
        for i in range(len(errors)-2, -1, -1):
            uhm = errors[i+1] @ self.layers[i+1].weights.T
            errors[i] = self.layers[i].activation.df(inbound[i]) * uhm
        return errors
    
    def calculate_grad(self, errors, inbound, outbound):
        batch_size = errors[0].shape[0]
        # todo: cleanup
        grad_b = [
            error.T.dot(np.ones((batch_size,1))).T/float(batch_size) 
            for error in errors
        ]
        
        inputs = [self.last_input] + outbound
        
        grad = [
            error.T.dot(inputs[i]).T/float(batch_size) 
            for i, error in enumerate(errors)
        ]
        return grad, grad_b
    
    def backpropagate(self, x, y):
        inbound, outbound = self.feedforward(x)
        errors = self.calculate_errors(y, inbound, outbound)
        grad, grad_b = self.calculate_grad(errors, inbound, outbound)
        return grad, grad_b
    
    def apply_grad(self, sumg, sumgb):
        for i in range(len(self.layers)):
            self.layers[i].weights += sumg[i]
            self.layers[i].bias += sumgb[i]
    
    def get_zero_grads(self):
        g0 = [g.weights * 0.0 for g in self.layers]
        gb0 = [gb.bias * 0.0 for gb in self.layers]
        return g0, gb0

    def __str__(self):
        from textwrap import indent 
        return "Layers(\n{}\n)".format(
            indent(
                "\n".join(
                    str(i) + ": " + str(l)
                    for i, l in enumerate(self.layers)
                ),
                "  "
            )
        )

class NNFactory:
    def __init__(self, activation_strategy: ActivationStrategy, layer_factory: LayerFactory):
        self.activation_strategy = activation_strategy
        self.layer_factory = layer_factory

    def get(self, architecture):
        layer_shapes = list(zip(architecture[:-1], architecture[1:]))
        activation_types = self.activation_strategy.execute(architecture)

        layers = [
            self.layer_factory.get(
                layer_shape,
                activation
            ) for layer_shape, activation 
            in zip(layer_shapes, activation_types)
        ]

        return NN(layers)

class ProgressTracker:
    def __init__(self, interval):
        self.interval = interval
        self.moments = []
        self.records = []

    def start(self, yhat, y):
        raise NotImplementedError

    def update(self, yhat, y):
        raise NotImplementedError

class DummyProgressTracker(ProgressTracker):
    def __init__(self):
        super().__init__(float("inf"))

    def start(self, yhat, y):
        pass

    def update(self, yhat, y):
        pass

class EpochLossTracker(ProgressTracker):
    def __init__(self, interval, metric):
        super().__init__(interval)
        self.metric = metric
    
    def start(self, yhat, y):
        self.moments.append(0)
        self.records.append(self.metric(yhat, y))

    def update(self, yhat, y):
        self.moments.append(self.moments[-1] + self.interval)
        self.records.append(self.metric(yhat, y))

class TimeLossTracker(ProgressTracker):  
    def __init__(self, interval, metric):
        super().__init__(interval)
        self.metric = metric
        self.start_time = None
    
    def start(self, yhat, y):
        self.start_time = time.time_ns()
        self.moments.append(0)
        self.records.append(self.metric(yhat, y))
    
    def update(self, yhat, y):
        loss = self.metric(yhat, y)
        moment = (time.time_ns() - self.start_time) * 1e-9
        print(f"\r{self.metric.__name__} at {moment:.2f}s: {loss}".ljust(30), end='')
        self.moments.append(moment)
        self.records.append(loss)

class Trainer:
    def __init__(self, nn: NN, x: np.array, y: np.array, x_test=None, y_test=None, tracker: ProgressTracker=None):
        self.nn = nn
        self.x = x
        self.y = y     
        self.x_test = x_test
        self.y_test = y_test
        self.tracker = tracker if tracker else DummyProgressTracker()

    def apply_rate(self, grads, rate):
        return [- g * rate for g in grads]

    def batch_descent(self, x, y, rate):
        sumg, sumgb = self.nn.backpropagate(x, y)
        self.nn.apply_grad(
            self.apply_rate(sumg,  rate),
            self.apply_rate(sumgb, rate),
        )
    
    def gradient_descent(self, rate):
        self.batch_descent(self.x, self.y, rate)

    def train_generic(self, method, max_epoch):
        self.tracker.start(self.nn.apply(self.x), self.y)
        for i in range(max_epoch):
            method()
            if i % self.tracker.interval == 0:
                self.tracker.update(self.nn.apply(self.x), self.y)

    def train_gradient(self, max_epoch, rate=1e-3):
        def gradient():
            self.gradient_descent(rate)
        self.train_generic(gradient, max_epoch)

    def train_random_batch(self, max_epoch, batch_size, rate=1e-3):
        def random_batch():
            indexes = np.random.randint(self.x.shape[0], size=batch_size)
            x_chosen = self.x[indexes]
            y_chosen = self.y[indexes]
            self.batch_descent(x_chosen, y_chosen, rate)
        self.train_generic(random_batch, max_epoch)

    def momentum_method_batch(self, max_epoch, batch_size=32, momentum_rate=0.9, rate=1e-3):
        m_g, m_gb = self.nn.get_zero_grads()
        def momentum_batch():
            nonlocal m_g, m_gb
            indexes = np.random.randint(self.x.shape[0], size=batch_size)
            x_chosen = self.x[indexes]
            y_chosen = self.y[indexes]
            g, gb = self.nn.backpropagate(x_chosen, y_chosen)
            m_g  = [mgi  * momentum_rate - rate * gi  for mgi, gi   in zip(m_g, g)]
            m_gb = [mgbi * momentum_rate - rate * gbi for mgbi, gbi in zip(m_gb, gb)]
            self.nn.apply_grad(m_g, m_gb)
        self.train_generic(momentum_batch, max_epoch)
    
    def rmsprop_method(self, max_epoch, slowdown_rate=0.9, rate=1e-3):
        eps = 0.0001
        slow_g , slow_gb = self.nn.get_zero_grads()
        def rmsprop():
            nonlocal slow_g , slow_gb
            g, gb = self.nn.backpropagate(self.x, self.y)
            
            slow_g = [slowgi * slowdown_rate + (1-slowdown_rate)*gi*gi
                for slowgi, gi in zip(slow_g, g)]

            slow_gb = [slowgbi * slowdown_rate + (1-slowdown_rate)*gbi*gbi
                for slowgbi, gbi in zip(slow_gb, gb)]

            final_g = [- rate * gi / np.sqrt(slowgi + eps) 
                for slowgi, gi in zip(slow_g, g)]

            final_gb = [- rate * gbi / np.sqrt(slowgbi + eps) 
                for slowgbi, gbi in zip(slow_gb, gb)]

            self.nn.apply_grad(final_g, final_gb)
        self.train_generic(rmsprop, max_epoch)

    def get_xy(self, dataset):
        if dataset == "train":
            return self.x, self.y
        elif dataset == "test":
            return self.x_test, self.y_test
        else:
            raise ValueError

    def evaluate(self, dataset="train"):
        x, y = self.get_xy(dataset)
        return MSE(self.nn.apply(x), y)

    def plot_results(self, title="", dataset="train"):
        x, y = self.get_xy(dataset)
        print(self.evaluate(dataset))
        plt.scatter(
            x, self.nn.apply(x)
        )
        plt.scatter(
            x, y
        )
        plt.title(title + ", " + dataset)
        plt.legend(["yhat", "y"])
        plt.show()
    
    def plot_progress(self):
        print(self.evaluate("train"))
        plt.plot(
            self.tracker.moments,
            self.tracker.records
        )

