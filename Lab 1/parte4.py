import sys
import sklearn
import torch
from torchvision import transforms
from torch.utils import data
from torch import nn
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
import parte3
import numpy as np

class MLPencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(784, 100)
            self.out = nn.Linear(100, 30)
        
        def forward(self, X):
            X = torch.flatten(X, start_dim=1)
            return self.out(F.relu(self.hidden(X)))

class MLPdecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(30, 100)
            self.out = nn.Linear(100, 784)
        
        def forward(self, X):
            X = F.relu(self.hidden(X))
            X = self.out(X)
            return X.view(-1, 28, 28)
        
class MLPautoencoder(nn.Module):
        def __init__(self, noise, encoder, decoder):
            super().__init__()
            self.noise = noise
            self.hidden = encoder
            self.out = decoder
        
        def forward(self, X):
            X = self.noise(X)
            X = self.hidden(X)
            X = self.out(X)
            return X

def test_loss(net, test_iter, loss):
    L, N = 0.0, 0
    for X, _ in test_iter:
        y_hat = net(X)
        l = loss(y_hat, X)
        L += l.sum().item()
        N += l.numel()
    return L / N

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
    L, N = 0.0, 0 
    for X, y in train_iter:
        l = loss(net(X), X.squeeze(1))
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        L += l.sum()
        N += l.numel()
    return L/N, None

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    metrics = []
    for epoch in range(num_epochs):
        L, _ = train_epoch(net, train_iter, loss, updater)
        TestLoss = test_loss(net, test_iter, loss)
        metric = (epoch + 1, L, TestLoss)
        print(f"Época {epoch + 1}: Pérdida entrenamiento={L:.4f}, Pérdida prueba={TestLoss:.4f}")
        metrics.append(metric)
    return metrics

def plot_reconstructions(model, images, n_images=10):
    noise = torch.nn.Sequential(torch.nn.Dropout(0.5))
    noise.train()
    input = noise(images)
    noise.eval()
    model.eval()
    reconstructions = np.clip(input[:n_images].squeeze().detach(), 0, 1)
    reconstructions = model(reconstructions).squeeze().detach()
    fig = plt.figure(figsize=(n_images * 2, 4))
    for image_index in range(n_images):
        plt.subplot(3, n_images, 1 + image_index)
        plt.imshow(images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plt.imshow(input[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + 2 * n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")

if __name__== "__main__":
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    torch.manual_seed(42)  # fijamos la semilla para generar reproducibilidad
    batch_size = 256

    # Dataloader para FashionMNIST
    mnist_train = torchvision.datasets.FashionMNIST(transform=transforms.ToTensor(),
            root="../data", train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(transform=transforms.ToTensor(),
            root="../data", train=False, download=True)
    iter_train, iter_valid =  (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=2),
                data.DataLoader(mnist_test, batch_size, shuffle=True,
                                num_workers=2))

    p=0.5 #probabilidad de que un pixel sea eliminado
    noise = torch.nn.Sequential(torch.nn.Dropout(p))

    images,_ = next(iter(iter_train))
    noise_images = noise(images)

    n_images = 10
    fig = plt.figure(figsize=(n_images * 2, 4))
    for image_index in range(n_images):
            plt.subplot(2, n_images, 1 + image_index)
            plt.imshow(images[image_index].squeeze(),
                    cmap="binary")
            plt.axis("off")
            plt.subplot(2, n_images, 1 + n_images + image_index)
            plt.imshow(noise_images[image_index].squeeze(),
                    cmap="binary")
            plt.axis("off")
    #plt.show()

    #Ejercicio 1
    print("-------Ejercicio 1------")
    encoder = MLPencoder()
    print("-------Test 1-1------")
    #@title Test N° 1
    #@markdown Ejecutar para confirmar que su código es correcto
    images,_ = next(iter(iter_train))
    try:
        latentes = encoder(images)
        assert latentes.shape[1] == 30, "La salida de su red no es un vector de 30 elementos"
        print("Al parecer está todo bien. Puedes avanzar al siguiente test")
    except:
        print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")

    print("-------Test 2-1------")     
    #@title Test N° 2
    #@markdown Ejecutar para confirmar que su código es correcto
    assert encoder.hidden.weight.shape == torch.Size([100, 784]), "Las dimensiones de su primera capa densa están mal"
    assert encoder.out.weight.shape == torch.Size([30, 100]), "Las dimensiones de su segunda capa densa están mal"
    print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")


    #Ejercicio 2
    print("-------Ejercicio 2------")
    decoder = MLPdecoder()
    print("-------Test 1-2------")
    #@title Test N° 1
    #@markdown Ejecutar para confirmar que su código es correcto
    try:
        salidas = decoder(latentes)
        assert salidas.shape[1] == 28 and salidas.shape[2] == 28, "La salida de su red no es una imagen de 28*28"
        print("Al parecer está todo bien. Puedes avanzar al siguiente test")
    except:
        print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")

    #@title Test N° 2
    print("-------Test 2-2------")    
    #@markdown Ejecutar para confirmar que su código es correcto
    assert decoder.hidden.weight.shape == torch.Size([100, 30]), "Las dimensiones de su primera capa densa están mal"
    assert decoder.out.weight.shape == torch.Size([784, 100]), "Las dimensiones de su segunda capa densa están mal"
    print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")

    #Ejercicio 3
    print("-------Ejercicio 3------")
    net = MLPautoencoder(noise, encoder, decoder)
    print("-------Test 1-3------")
    #@title Test N° 1
    #@markdown Ejecutar para confirmar que su código es correcto
    try:
        salidas = net(images)
        assert salidas.size == salidas.size, "La salida de su red no tiene el mismo tamaño que la entrada"
        print("Al parecer está todo bien. Puedes avanzar al siguiente test")
    except:
        print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")
    
    print("-------Test 2-3------")    
    #@title Test N° 2
    #@markdown Ejecutar para confirmar que su código es correcto
    assert net.noise == noise, "Tu primer bloque no es el correcto"
    assert net.hidden == encoder, "Tu segundo bloque no es el correcto"
    assert net.out == decoder, "Tu tercer bloque no es el correcto"
    print("Al parecer está todo bien. Puedes avanzar al siguiente test")
 
    #Ejercicio 4
    print("-------Ejercicio 4------")    
    loss = torch.nn.MSELoss()
    trainer = torch.optim.Adam(net.parameters())
    net.train()
    #ingrese su código aquí
    num_epoch = 5
    metrics = train(net, iter_train, iter_valid, loss, num_epoch, trainer)
    #@title Grafique Predicciones de Validación
    # Codigo adicional para generar imágenes.
    net.eval()
    plot_reconstructions(net, next(iter(iter_valid))[0])
    plt.show()






