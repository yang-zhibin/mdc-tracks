import torch
import torch.nn.functional as F

torch.manual_seed(0)

#import lightning.pytorch as pl

class DriftDistanceRegression(torch.nn.Module):
    def __init__(self):
        super(DriftDistanceRegression, self).__init__()

        self.fc1 = torch.nn.Linear(1, 64)  
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 1)
          
        


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out)) 
        out = F.relu(self.fc3(out)) 
        out= self.fc4(out)
        return out

class PolynomialRegression(torch.nn.Module):
    def __init__(self, degree=2):
        super(PolynomialRegression, self).__init__()
        self.degree = degree
        self.poly = torch.nn.Linear(degree + 1, 2048)
        self.linear = torch.nn.Linear(2048, 1)

    def forward(self, x):
        # Expand input to polynomial terms
        x_poly = x.new_zeros(x.size()[0], self.degree + 1)
        for d in range(self.degree + 1):
            x_poly[:, d] = x.squeeze(1)

        # Compute output using polynomial layer
        y_pred = self.poly(x_poly)
        y_pred = F.relu(y_pred)
        y_pred = self.linear(y_pred)
        
        return y_pred