# extraordinary simple red blood cell model using graph neural networks

## model loading and euler dif equation solver test

![](doc/images/random_force.gif)


- sphere model is loaded from obj file (86 polygons), using **libs_common.obj_model**
- obj model is converted to graph NN daata format, using **libs_common.physical_model**
- random force is applied on model
- after 500 steps, model is reloaded and add new noise from randomizer
- and result is rendered

the Euler solver is solving 2nd order dif eqautions (from Newtons law F=ma)
```
v(n+1) = v(n) + f(n) dt #velocity is integral of force/mass (unit mass used)
x(n+1) = x(n) + v(n) dt #position is integral of velocity
```
implementation is in **libs_common.physical_model**, 
note : lines v(n+1) and x(n+1) are swapped in code, for better stability

```python
#./rbc_model_test.py
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
```


## dependences

```bash
$pip3 install numpy torch torch_geometric torch_sparse torch_scatter PyOpenGL glfw
```
