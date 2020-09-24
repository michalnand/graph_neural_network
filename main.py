import libs_common
import torch
import numpy

import models.rbc_model.model


def rnd(min, max):
    return numpy.random.randint(max - min) + min

#load obj model
obj_model = libs_common.ObjModel("./models_obj/sphere_86.obj")
#obj_model = libs_common.ObjModel("./models_obj/sphere_960.obj")

#random noise for position or velocity
randomizer = libs_common.Randomizer(sigma_position = 0.05, sigma_velocity = 0.02)

#create data
data = libs_common.PhysicalModel(obj_model.points, obj_model.polygons, randomizer=randomizer) 


Model = models.rbc_model.model


trainer = libs_common.TrainModel(data, Model, learning_rate = 0.01) 

epoch_count = 100

steps_min   = 256
steps_max   = 1024

for epoch in range(epoch_count):
    steps_  = rnd(steps_min, steps_max)
    loss    = trainer.process_trajectory(steps=steps_, dt = 0.01)
    print(epoch, loss)


#visualisation
window = libs_common.RenderModel()


steps       = 1024

step = 0
while True:
    if step%steps == 0:
        #apply randomizer
        data.reset()   

    #generate random force
    force = trainer.model(data.position, data.velocity, data.force, data.edge_index).detach()

    #proces Euler solver
    data.step(force)

    print("center_of_mass = ", data.get_center_of_mass())
    print("volume = ", data.get_volume())
    print("\n\n\n")

    #render output using opengl
    points          = data.position.to("cpu").detach().numpy()
    points_initial  = data.position_initial.to("cpu").detach().numpy()
    
    edges  = data.edge_index.to("cpu").detach().numpy()
    window.render(points, points_initial, edges)

    step+= 1
