import time
import numpy as np
from pathlib import Path
import coal
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

"""
TODO: 
- remove self collisions
- display robot semi transparent
- refine the obstacles definitions
"""


model_dir = Path("robot_model")
urdf_path = model_dir / "panda_robotiq.urdf"
# srdf_path = model_dir / "panda_robotiq.srdf"
srdf_path = model_dir / "panda_robotiq_mod.srdf"
# rmodel: pin.Model, contains information about the robot kinematic chain (frames, joints), intertias, etc.
# cmodel: pin.GeometryModel, contains information about the robot links geometry, used for collision detection (simplified shapes)
# cmodel: pin.GeometryModel, contains information about the robot links geometry, used for visualizations
rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(urdf_path, package_dirs=model_dir)
print("rmodel.nq", rmodel.nq)

# make the robot slightly transparent
for go in cmodel.geometryObjects:
    go.meshColor = np.array((1,1,1,0.5))

# some test configurations
# panda_robotiq model has 13 dofs, 7 for panda + 6 for robotiq
# luckily, 0 angle for robotiq configuration corresponds to open gripper
# we'll use that for now as it is a good upper bound
# q_panda = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])  # canonical
q_panda = np.array([0,  0.7, 0, -2.35619449019, 0, 2.0, 0.785398163397])  # in table collision
q_robotiq = np.zeros(rmodel.nq - 7)
# q_robotiq = -0.5*np.ones(rmodel.nq - 7)
q = np.hstack([q_panda, q_robotiq])

# Add obstacles defined by (name, coal.CollisionGeometry, t_obs, R_robs, parent_joint_id, color)
obstacles = [
    ("ground", coal.Box(1.2,1,1.0), np.array([0.4,0,-0.5]), np.eye(3), 0, (1,0,0,0.5)),
    ("cam_beam_1", coal.Box(0.1,0.1,4), np.array([-0.2,-0.5,0]), np.eye(3), 0, (1,0,0,0.5)),
]

for obs_name, obs_coal, t_obs, R_obs, parent_joint_id, color in obstacles:
    Mobs = pin.SE3(R_obs, t_obs)
    obs_id_frame = rmodel.addFrame(pin.Frame(obs_name, parent_joint_id, Mobs, pin.OP_FRAME))  # try
    obs_geom = pin.GeometryObject(
        obs_name, parent_joint_id, obs_id_frame, rmodel.frames[obs_id_frame].placement, obs_coal
    )
    obs_geom.meshColor = np.array(color)
    cmodel.addGeometryObject(obs_geom)

# add all robot self collisions and remove the ones in srdf "disable_collisions" tags
cmodel.addAllCollisionPairs()
print("num collision pairs - after adding all:", len(cmodel.collisionPairs))
pin.removeCollisionPairs(rmodel, cmodel, srdf_path)
print(
    "num collision pairs - after removing disabled collision pairs:",
    len(cmodel.collisionPairs),
)


# geometry id to name mapping 
goid_to_name = [go.name for go in cmodel.geometryObjects]

# Create data structures
data = rmodel.createData()
collision_data = pin.GeometryData(cmodel)

# Compute all the collisions
t1 = time.time()
stop_at_first_collision = False
in_collision = pin.computeCollisions(rmodel, data, cmodel, collision_data, q, stop_at_first_collision)
print(f"pin.computeCollisions took {1000*(time.time() - t1)} ms")
pin.computeDistances(cmodel, collision_data)

# Print the status of collision for all collision pairs
collision_pair_contacts = []
pairs_in_collision = []
for k in range(len(cmodel.collisionPairs)):
    cr = collision_data.collisionResults[k]
    cp = cmodel.collisionPairs[k]
    if cr.isCollision():
        assert(len(cr.getContacts()) == 1)
        # print(f"Collision {cp.first} - {cp.second}")
        print(f"Collision {goid_to_name[cp.first]} - {goid_to_name[cp.second]}")
        # first cr gives always nan points (not sure why)
        contact = cr.getContact(0)
        print(contact.getNearestPoint1())  # always gives nans
        if np.isnan(contact.getNearestPoint1()[0]):
            continue

        pairs_in_collision.append(cp)
        collision_pair_contacts.append((
            contact.getNearestPoint1(), 
            contact.getNearestPoint2()
        ))


import meshcat
from meshcat.geometry import Sphere, MeshBasicMaterial, PointsGeometry, MeshBasicMaterial, Line

viz = MeshcatVisualizer(rmodel, cmodel, vmodel)
viz.initViewer(loadModel=True)
viz.displayVisuals(False)
viz.displayCollisions(True)
viz.displayFrames(False)
viz.display(q)
breakpoint()

for i in range(len(collision_pair_contacts)):
    sphere_size = 0.005

    pt1, pt2 = collision_pair_contacts[i]
    goid1, goid2 = collision_pair_contacts[i]
    
    # Create the two green dots
    viz.viewer[f"cpair/{i:03d}_p1"].set_object(
        Sphere(sphere_size),
        MeshBasicMaterial(color=0x00ff00)  # Green
    )
    viz.viewer[f"cpair/{i:03d}_p1"].set_transform(meshcat.transformations.translation_matrix(pt1))

    viz.viewer[f"cpair/{i:03d}_p2"].set_object(
        Sphere(sphere_size),
        MeshBasicMaterial(color=0x00ff00)  # Green
    )
    viz.viewer[f"cpair/{i:03d}_p2"].set_transform(meshcat.transformations.translation_matrix(pt2))

    # Create the blue line connecting them
    vertices = np.array((pt1, pt2)).T
    breakpoint()
    viz.viewer[f"cpair/line_{i:03d}"].set_object(
        Line(
            PointsGeometry(vertices),
            MeshBasicMaterial(color=0x0000ff)  # Blue
        )
    )



