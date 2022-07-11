import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random

import torch
import torch.nn as nn
from torch import float32, float64, int64, optim

if __name__ == "__main__":
    digits = load_digits()
    
    """
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:5]): # 5개의 샘플만 출력
        plt.subplot(2, 5, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('sample: %i' % label)
        plt.show()
    """
    
    model = nn.Sequential(
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,10) # Using softmax, and 
    )
    
    X = torch.tensor(digits.data,dtype=float32)
    Y = torch.tensor(digits.target,dtype=int64)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    losses = []
    for epoch in range(100):
        hypothesis = model(X)
        cost = loss_fn(hypothesis,Y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, cost.item()
            ))
            
        losses.append(cost.item())
        
    plt.plot(losses)
    plt.show()
    
    with torch.no_grad():
        X_test = X
        Y_test = Y
        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
        
        r = random.randint(0, len(digits.images) - 1)
        X_single_data = torch.tensor(digits.data[r:r+1]).float()
        Y_single_data = torch.tensor(digits.target[r:r+1],dtype=int64)
        
        print("label: ",Y_single_data.item())
        print(X_single_data.view(8,8))
        single_prediction = model(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, 1).item())
        
        plt.imshow(torch.tensor(digits.data[r:r+1]).view(8,8).float(),cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()