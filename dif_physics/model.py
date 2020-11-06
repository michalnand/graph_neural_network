import torch
import torch.nn as nn

import numpy
import matplotlib.pyplot as plt


class ThrowingSphereModel(torch.nn.Module):
    def __init__(self):
        super(ThrowingSphereModel, self).__init__()

        #model parameters, set some initial values
        self.angle      = nn.Parameter(torch.ones(1))
        self.velocity   = nn.Parameter(10.0*torch.ones(1))


    def forward(self, steps, dt):

        #initial position [m]
        x = 0.0
        y = 0.0

        #body mass, [kg]
        mass = 0.150
        
        #initial velocity convert into x, y coordinates
        vx = self.velocity*torch.cos(self.angle)
        vy = self.velocity*torch.sin(self.angle)

        #just for storing whole trajectory and ploting
        trajectory = numpy.zeros((2, steps))


        for step in range(steps):   

            
            
            #F = ma, newton 2nd law
            #x : there is no horizontal force
            #y : there is only gravity force
            fx = 0.0
            fy = -9.81*mass

            #velocity is integral of force
            vx+= fx*dt
            vy+= fy*dt

            #position is integral of velocity
            x+= vx*dt
            y+= vy*dt

            #store point
            trajectory[0][step] = x.detach().numpy()[0]
            trajectory[1][step] = y.detach().numpy()[0]

        return x, y, trajectory


def compute_loss(target_x, target_y, predicted_x, predicted_y):
    #compute loss, mean square error
    loss_x = (target_x - predicted_x)**2
    loss_y = (target_y - predicted_y)**2
    loss   = loss_x + loss_y
    return loss

#create model
model       = ThrowingSphereModel()

#create solver, leanring rate 0.1
optimizer   = torch.optim.Adam(model.parameters(), lr=0.1)

#some target position
target_x = 30.0
target_y = 10.0

epoch_count = 200

#process training
for epoch in range(epoch_count):
    #obtain final point
    x, y, trajectory = model.forward(steps=1024, dt=0.01)

    #compute loss, mean square error
    loss   = compute_loss(target_x, target_y, x, y)

    #clear previous gradients
    optimizer.zero_grad()

    #compute gradients
    loss.backward()

    #update model parameters
    optimizer.step()
    
    #print result
    print("epoch    = ", epoch)
    print("x_target = ", target_x)
    print("x        = ", x.detach().numpy())
    print("y_target = ", target_y)
    print("y        = ", y.detach().numpy())
    print("angle    = ", model.angle.detach().numpy())
    print("velocity = ", model.velocity.detach().numpy())
    print("\n\n\n")

    #plot result
    if epoch%10 == 0:
        plt.plot(trajectory[0], trajectory[1],  color='blue')
        plt.plot(target_x, target_y, 'rp', markersize=14)
        plt.draw()
        plt.pause(0.001)