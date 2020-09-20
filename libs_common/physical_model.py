import torch

class PhysicalModel:
    def __init__(self, points, polygons, position = [0, 0, 0], velocity = [0, 0, 0], randomizer = None, device = "cpu"):
        
        self.randomizer     = randomizer
        self.device         = device
        self.points_count   = len(points)

        self._compute_points(points, position, velocity)
        self._compute_edges(polygons)     

        self.reset()   

        self.in_features_count  = 3*3
        self.out_features_count = 3
    
        print("points_shape     = ",  self.x.shape)
        print("edges_shape      = ",  self.edge_index.shape)
        print("in_features_count= ",  self.in_features_count)
        print("out_features_count= ", self.out_features_count)


    def _compute_points(self, points, position, velocity):
        self.x_initial = torch.zeros((self.points_count, 4, 3), dtype=torch.float)
        self.x_initial.to(self.device)
        
        for i in range(self.points_count):
            self.x_initial[i][0][0] = points[i][0]
            self.x_initial[i][0][1] = points[i][1]
            self.x_initial[i][0][2] = points[i][2]

            self.x_initial[i][1][0] = velocity[0]
            self.x_initial[i][1][1] = velocity[1]
            self.x_initial[i][1][2] = velocity[2]

            self.x_initial[i][2][0] = 0.0
            self.x_initial[i][2][1] = 0.0
            self.x_initial[i][2][2] = 0.0

    def reset(self):
        self.x = self.x_initial.clone()

        if self.randomizer is not None:
            self.randomizer.next()
 
            for i in range(self.points_count):
                position_noise = self.randomizer.get_position()
                velocity_noise = self.randomizer.get_velocity()

                self.x[i][0][0]+= position_noise[0]
                self.x[i][0][1]+= position_noise[1]
                self.x[i][0][2]+= position_noise[2]

                self.x[i][1][0]+= velocity_noise[0]
                self.x[i][1][1]+= velocity_noise[1]
                self.x[i][1][2]+= velocity_noise[2]

    def step(self, force, dt = 0.01):
        self.x = torch.transpose(self.x, 1, 0)

        self.x[2] = force.clone()
        self.x[1] = self.x[1] + self.x[2]*dt
        self.x[0] = self.x[0] + self.x[1]*dt
 
        self.x = torch.transpose(self.x, 0, 1)
 
    def get_center_of_mass(self):

        x = torch.transpose(self.x, 1, 0)

        position = x[0].mean(dim=0)
        velocity = x[1].mean(dim=0)
        force    = x[2].mean(dim=0)

        return position, velocity, force


    def get_volume(self):
        center, _, _ = self.get_center_of_mass()

        position = torch.transpose(self.x, 1, 0)[0]

        position = position - center

        v = torch.norm(position, dim=1)**3

        volume = v.mean()

        return volume

    def get_surface(self):
        #TODO
        return 0

    def get_curvature(self):
        #TODO
        return 0



    def _compute_edges(self, polygons):
        starting_points = []
        ending_points   = []

        for p in polygons:
            starting_points.append(p[0])
            ending_points.append(p[1])
            starting_points.append(p[0])
            ending_points.append(p[2])

            starting_points.append(p[1])
            ending_points.append(p[0])
            starting_points.append(p[1])
            ending_points.append(p[2])

            starting_points.append(p[2])
            ending_points.append(p[0])
            starting_points.append(p[2])
            ending_points.append(p[1])

        self.edge_index = torch.tensor([starting_points, ending_points], dtype=torch.long)
        self.edge_index = self.edge_index.to(self.device)
