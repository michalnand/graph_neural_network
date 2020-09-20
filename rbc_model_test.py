import libs_common
import torch

#load obj model
obj_model = libs_common.ObjModel("./models_obj/sphere_86.obj")
#obj_model = libs_common.ObjModel("./models_obj/sphere_960.obj")

#random noise for position or velocity
randomizer = libs_common.Randomizer(sigma_position = 0.03, sigma_velocity = 0.03)

#create data
data = libs_common.PhysicalModel(obj_model.points, obj_model.polygons, randomizer=randomizer) 

#visualisation
window = libs_common.RenderModel()


steps = 0

while True:
    if steps%500 == 0:
        #apply randomizer
        data.reset()   

    #generate random force
    force  = torch.randn(data.points_count, 3)

    #proces Euler solver
    data.step(force)

    #render output using opengl
    points = data.x.to("cpu").detach().numpy()
    edges  = data.edge_index.to("cpu").detach().numpy()
    window.render(points, edges)

    steps+= 1