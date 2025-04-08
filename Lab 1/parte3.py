import torch
import torchvision
from IPython import display
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
from torch import nn

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=1))

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    '''
        Lleva adelante el entrenamiento completo llamando a funciones internas
        que modularizan el ciclo de entrenamiento.

        Parámetros:
                net: la red neuronal que se va a entrenar
                train_iter: iterador de datos de entrenamiento
                test_iter: iterador de datos de prueba
                loss: función de perdida a minimizar
                num_epoch: cantidad de épocas a entrenar
                updater: algoritmo de optimización

        Salida:
                metrics: una lista de tuplas (una para cada epoch)
                    con las siguientes componentes
                    - epoch: número de época
                    - L: pérdida calculada
                    - Acc: accuracy de entrenamiento calculada
                    - TestAcc: accuracy de prueba calculada
    '''
    metrics =[]
    for epoch in range(num_epochs):
        L, Acc = train_epoch(net, train_iter, loss, updater)
        TestAcc = test_accuracy(net, test_iter)
        metric = (epoch + 1, L, Acc, TestAcc)
        print(metric)
        metrics.append(metric)
    return metrics


def train_epoch(net, train_iter, loss, updater):
    '''
        Lleva adelante el entrenamiento de una sola época.

        Parámetros:
                net: la red neuronal que se va a entrenar
                train_iter: iterador de datos de entrenamiento
                loss: función de perdida a minimizar
                updater: algoritmo de optimización

        Salida:
                L: pérdida calculada
                Acc: accuracy de entrenamiento calculada
    '''
    # inserte su código aquí
    L, Acc, N = 0.0, 0.0, 0 
    for X, y in train_iter:
        l = loss(net(X)   ,y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        L += l.sum()
        N += l.numel()
        Acc += accuracy(net(X), y)
    return L/N, Acc/N

def test_accuracy(net, test_iter):
    '''
    Evalúa los resultados del entrenamiento de una sola época.

    Parámetros:
            net: la red neuronal que se va a evaluar
            test_iter: iterador de datos de prueba

    Salida:
            - TestAcc: accuracy de prueba calculada
    '''
    # inserte su código aquí
    N, Acc = 0, 0.0
    for X, y in test_iter:
        Acc += accuracy(net(X), y)
        N += y.numel()
    return Acc/N

if __name__ == "__main__": 
    '''
    #Ejercicio 1
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    net1 = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net1.apply(init_weights)
    num_batch = 256
    train_iter, test_iter = load_data_fashion_mnist(num_batch)

    
    
    #Ejercicio 2
    lr = 0.3
    trainer = torch.optim.SGD(net1.parameters(), lr)
    num_epoch = 10
    print("-------------Ejercicio 2---------------")
    metrics1 = train(net1, train_iter, test_iter, loss, num_epoch, trainer)
    
    epochs1 = [int(m[0]) for m in metrics1]
    train_loss1 = [float(m[1]) for m in metrics1]
    train_acc1 = [float(m[2]) for m in metrics1]
    test_acc1 = [float(m[3]) for m in metrics1]

    #Graficar
    plt.figure(figsize=(10, 6))

    plt.plot(epochs1, train_loss1, 'r-', label='Pérdida de entrenamiento')
    plt.plot(epochs1, train_acc1, 'b-o', label='Accuracy de entrenamiento')
    plt.plot(epochs1, test_acc1, 'g-s', label='Accuracy de prueba')

    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Evolución de pérdida y accuracy durante el entrenamiento ")
    plt.legend("Gráficas ejercicio 2")
    plt.grid(True)
    plt.show()
    '''
    
    '''
    #Ejercicio 3
    lr = 0.3
    trainer = torch.optim.SGD(net1.parameters(), lr)
    num_epoch = 20
    print("-------------Ejercicio 3---------------")
    metrics2 = train(net1, train_iter, test_iter, loss, num_epoch, trainer)
    
    epochs2 = [int(m[0]) for m in metrics2]
    train_loss2 = [float(m[1]) for m in metrics2]
    train_acc2 = [float(m[2]) for m in metrics2]
    test_acc2 = [float(m[3]) for m in metrics2]
    # Graficar
    plt.figure(figsize=(20, 6))
    plt.plot(epochs2, train_loss2, 'r-', label='Pérdida de entrenamiento')
    plt.plot(epochs2, train_acc2, 'b-o', label='Accuracy de entrenamiento')
    plt.plot(epochs2, test_acc2, 'g-s', label='Accuracy de prueba')

    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Evolución de pérdida y accuracy durante el entrenamiento")
    plt.legend("Gráficas ejercicio 3")
    plt.grid(True)
    plt.show()

    #Ejercicio 4
    lr = 1
    trainer = torch.optim.SGD(net1.parameters(), lr)
    num_epoch = 20
    print("-------------Ejercicio 4---------------")
    metrics3 = train(net1, train_iter, test_iter, loss, num_epoch, trainer)
    
    epochs3 = [int(m[0]) for m in metrics3]
    train_loss3 = [float(m[1]) for m in metrics3]
    train_acc3 = [float(m[2]) for m in metrics3]
    test_acc3 = [float(m[3]) for m in metrics3]
    # Graficar
    plt.figure(figsize=(20, 6))
    plt.plot(epochs3, train_loss3, 'r-', label='Pérdida de entrenamiento')
    plt.plot(epochs3, train_acc3, 'b-o', label='Accuracy de entrenamiento')
    plt.plot(epochs3, test_acc3, 'g-s', label='Accuracy de prueba')

    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Evolución de pérdida y accuracy durante el entrenamiento")
    plt.legend("Gráficas ejercicio 4")
    plt.grid(True)
    plt.show()
    
    #Ejercicio 5

    loss = torch.nn.CrossEntropyLoss(reduction='none')
    net2 = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.Sigmoid(),
                        nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net2.apply(init_weights)
    num_batch = 256
    train_iter, test_iter = load_data_fashion_mnist(num_batch)

    lr = 1
    trainer = torch.optim.SGD(net2.parameters(), lr)
    num_epoch = 20
    print("-------------Ejercicio 5---------------")
    metrics4 = train(net2, train_iter, test_iter, loss, num_epoch, trainer)
    
    epochs4 = [int(m[0]) for m in metrics4]
    train_loss4 = [float(m[1]) for m in metrics4]
    train_acc4 = [float(m[2]) for m in metrics4]
    test_acc4 = [float(m[3]) for m in metrics4]
    # Graficar
    plt.figure(figsize=(20, 6))
    plt.plot(epochs4, train_loss4, 'r-', label='Pérdida de entrenamiento')
    plt.plot(epochs4, train_acc4, 'b-o', label='Accuracy de entrenamiento')
    plt.plot(epochs4, test_acc4, 'g-s', label='Accuracy de prueba')

    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Evolución de pérdida y accuracy durante el entrenamiento")
    plt.legend("Gráficas ejercicio 4")
    plt.grid(True)
    plt.show()

    '''

    #Ejercicio 6
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    net3 = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 1024),
                        nn.Sigmoid(),
                        nn.Linear(1024, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net3.apply(init_weights)
    num_batch = 256
    train_iter, test_iter = load_data_fashion_mnist(num_batch)

    lr = 1
    trainer = torch.optim.SGD(net3.parameters(), lr)
    num_epoch = 20
    print("-------------Ejercicio 6---------------")
    metrics4 = train(net3, train_iter, test_iter, loss, num_epoch, trainer)
    
    epochs5 = [int(m[0]) for m in metrics4]
    train_loss5 = [float(m[1]) for m in metrics4]
    train_acc5 = [float(m[2]) for m in metrics4]
    test_acc5 = [float(m[3]) for m in metrics4]
    # Graficar
    plt.figure(figsize=(20, 6))
    plt.plot(epochs5, train_loss5, 'r-', label='Pérdida de entrenamiento')
    plt.plot(epochs5, train_acc5, 'b-o', label='Accuracy de entrenamiento')
    plt.plot(epochs5, test_acc5, 'g-s', label='Accuracy de prueba')

    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Evolución de pérdida y accuracy durante el entrenamiento")
    plt.legend("Gráficas ejercicio 4")
    plt.grid(True)
    plt.show()