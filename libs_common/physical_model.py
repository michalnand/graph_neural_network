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
    
        print("position_shape     = ",  self.position.shape)
        print("edges_shape      = ",  self.edge_index.shape)
        print("in_features_count= ",  self.in_features_count)
        print("out_features_count= ", self.out_features_count)


    def _compute_points(self, points, position, velocity):
        self.position_initial = torch.zeros((self.points_count, 3), dtype=torch.float)
        self.position_initial.to(self.device)

        self.velocity_initial = torch.zeros((self.points_count, 3), dtype=torch.float)
        self.velocity_initial.to(self.device)

        self.force_initial = torch.zeros((self.points_count, 3), dtype=torch.float)
        self.force_initial.to(self.device)
        
        for i in range(self.points_count):
            self.position_initial[i][0] = points[i][0]
            self.position_initial[i][1] = points[i][1]
            self.position_initial[i][2] = points[i][2]

            self.velocity_initial[i][0] = velocity[0]
            self.velocity_initial[i][1] = velocity[1]
            self.velocity_initial[i][2] = velocity[2]

            self.force_initial[i][0] = 0.0
            self.force_initial[i][1] = 0.0
            self.force_initial[i][2] = 0.0

    def reset(self):
        self.position   = self.position_initial.clone()
        self.velocity   = self.velocity_initial.clone()
        self.force      = self.force_initial.clone()

        if self.randomizer is not None:
            self.randomizer.next()

            scale = 1.0 #self.randomizer.get_scale()
 
            for i in range(self.points_count):
                position_noise = self.randomizer.get_position()
                velocity_noise = self.randomizer.get_velocity()

                self.position[i][0] = (self.position[i][0] + position_noise[0])*scale
                self.position[i][1] = (self.position[i][1] + position_noise[1])*scale
                self.position[i][2] = (self.position[i][2] + position_noise[2])*scale
                
                self.velocity[i][0] = (self.velocity[i][0] + velocity_noise[0])*scale
                self.velocity[i][1] = (self.velocity[i][1] + velocity_noise[1])*scale
                self.velocity[i][2] = (self.velocity[i][2] + velocity_noise[2])*scale
                 
    def step(self, force, dt = 0.01):
        self.velocity = self.velocity + force*dt
        self.position = self.position + self.velocity*dt

 
    def get_center_of_mass(self):

        position = self.position.mean(dim=0)
        velocity = self.velocity.mean(dim=0)
        force    = self.force.mean(dim=0)

        return position, velocity, force


    def get_volume(self):

        center, _, _ = self.get_center_of_mass()

        position = self.position - center.detach()

        v = torch.norm(position, dim=1)**3

        volume = v.mean()


        return volume

    def get_surface(self):
        #TODO
        return 0

    #https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
    def get_curvature(self):
        center, _, _ = self.get_center_of_mass()

        edges_count = self.edge_index.shape[1]

        curvature = torch.zeros(edges_count).to(self.device)

        for i in range(edges_count):  
            pa_idx = self.edge_index[0][i]
            pb_idx = self.edge_index[1][i]

            pa = self.position[pa_idx]
            pb = self.position[pb_idx]

            na = self.position[pa_idx] - center
            nb = self.position[pb_idx] - center

            print(pa - pb, na - nb)

            curvature[i] = torch.norm(pa - pb)/torch.norm(na - nb)


        return curvature.mean()



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
