import libs_common
import torch

from torchviz import make_dot


class TrainModel:
    def __init__(self, data, Model, learning_rate = 0.01, weight_decay = 0.00001):
        
        self.data           = data
        in_features_count   = self.data.in_features_count
        out_features_count  = self.data.out_features_count
        
        self.model          = Model.Create(in_features_count, out_features_count, hidden_count = 64)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)


    def process_trajectory(self, steps = 10, dt = 0.01):
        
        self.data.reset()   
        

        for step in range(steps):
            #compute force

            position, velocity, force = self.data.get_center_of_mass()

            position_   = self.data.position    - position
            velocity_   = self.data.velocity    - velocity
            force_      = self.data.force       - force

            force = self.model(position_, velocity_, force_, self.data.edge_index)

            #proces Euler solver
            self.data.step(force)

        #target volume
        loss_volume     = (0.3 - self.data.get_volume())**2

        #loss_curvature  = self.data.get_curvature()

        loss_velocity   = ((self.data.velocity)**2).mean()

        
        loss = loss_volume  + loss_velocity

        #make_dot(loss).render("model", format="png")
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_np = loss.detach().to("cpu").numpy()

        print(">>>> ", loss_np, self.data.get_volume())

        return loss_np

