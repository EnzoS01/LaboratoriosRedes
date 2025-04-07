import torch
import torchvision
from IPython import display
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

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

  net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
  net.apply(init_weights)

  loss = torch.nn.CrossEntropyLoss(reduction='none')
  trainer = torch.optim.SGD(net.parameters(), lr=0.1)
  

  #inserte su código aquí
  num_epoch = 10
  num_batch = 256
  train_iter, test_iter = load_data_fashion_mnist(num_batch)
  metrics = train(net, train_iter, test_iter, loss, num_epoch, trainer)

  #inserte su código aquí
  epochs = [int(m[0]) for m in metrics]
  train_loss = [float(m[1]) for m in metrics]
  train_acc = [float(m[2]) for m in metrics]
  test_acc = [float(m[3]) for m in metrics]

  # Graficar

  plt.figure(figsize=(10, 6))

  plt.plot(epochs, train_loss, 'r-', label='Pérdida de entrenamiento')
  plt.plot(epochs, train_acc, 'b-o', label='Accuracy de entrenamiento')
  plt.plot(epochs, test_acc, 'g-s', label='Accuracy de prueba')

  plt.xlabel("Épocas")
  plt.ylabel("Valor")
  plt.title("Evolución de pérdida y accuracy durante el entrenamiento")
  plt.legend()
  plt.grid(True)
  plt.show()

