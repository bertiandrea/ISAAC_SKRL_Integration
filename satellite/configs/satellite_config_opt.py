# satellite_config.py

from satellite.configs.base_config import BaseConfig

from pathlib import Path
import numpy as np

NUM_ENVS = 8
ROLLOUTS = 8

class SatelliteConfigOptimization(BaseConfig):
    set_seed = False
    seed = 42

    physics_engine = 'physx'

    rl_device="cuda:0"
    sim_device="cuda:0"
    graphics_device_id=0
    headless=True
    virtual_screen_capture=False
    force_render=False

    profile = False
    
    heartbeat = False
    
    class env:  
        numEnvs = NUM_ENVS
   
        numObservations = 14 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az, actX,actY,actZ]

        numStates = 17 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az, actX,actY,actZ, vx,vy,vz]

        numActions = 3
        
        envSpacing = 2.0

        sensor_noise_std = 0.0
        actuation_noise_std = 0.0
        
        threshold_ang_goal = 0.0872665        # soglia in radianti per orientamento
        threshold_vel_goal = 0.0174532        # soglia in rad/sec per la differenza di velocità
        overspeed_ang_vel =  0.78540        # soglia in rad/sec per l'overspeed
        episode_length_s = 30              # soglia in secondi per la terminazione di una singola simulazione
        
        clipActions = np.Inf
        clipObservations = np.Inf

        torque_scale = 1

        debug_arrows = False
        
        class asset:
            assetRoot = str(Path(__file__).resolve().parent.parent)
            assetFileName = "satellite.urdf"
            assetName = "satellite"

            init_pos_p = [0, 0, 0]    # posizione iniziale del satellite [x,y,z]
            init_pos_r = [0, 0, 0, 1] # attitude iniziale del satellite [x,y,z,w]

            #disable_gravity
            #collapse_fixed_joints
            #slices_per_cylinder
            #replace_cylinder_with_capsule
            #fix_base_link
            #default_dof_drive_mode
            #self_collisions
            #flip_visual_attachments

            #density
            #angular_damping
            #linear_damping
            #max_angular_velocity
            #max_linear_velocity
            #armature
            #thickness

    class sim:
        dt = 1.0 / 60.0
        gravity = [0.0, 0.0, 0.0] # [m/s^2]
        up_axis = 'z'
        use_gpu_pipeline = True
        substeps = 2
        
        #num_client_threads
        #stress_visualization
        #stress_visualization_max
        #stress_visualization_min
        
        class physx:
            use_gpu = True
            #solver_type = 1
            #num_threads = 4
            #num_position_iterations = 4
            #num_velocity_iterations = 1
            #contact_offset
            #rest_offset
            #bounce_threshold_velocity
            #contact_collection
            #default_buffer_size_multiplier
            #max_depenetration_velocity
            #max_gpu_contact_pairs
            #num_subscenes
            #always_use_articulations
            #friction_correlation_distance
            #friction_offset_threshold
            
        #class flex:
            #solver_type
            #num_outer_iterations
            #num_inner_iterations
            #relaxation
            #warm_start
            #contact_regularization
            #deterministic_mode
            #dynamic_friction
            #friction_mode
            #geometric_stiffness
            #max_rigid_contacts
            #max_soft_contacts
            #particle_friction
            #return_contacts
            #shape_collision_distance
            #shape_collision_margin
            #static_friction

    class rl:
        class PPO:
            num_envs = NUM_ENVS #Number of parallel environments collecting experience; more envs yield better GPU/utilization but higher memory use.
            rollouts = ROLLOUTS #Number of steps per environment before each policy update (i.e. rollout length).
            learning_epochs = 4 #How many times to iterate over the collected batch of data when updating the policy.
            mini_batches = 8 #Number of chunks to split the rollout batch into for stochastic gradient descent.
            
            class experiment:
                    write_interval = 1
                    checkpoint_interval = 1
                    directory = "./runs/satellite"
                    experiment_name = "satellite_test"
                    wandb = False

        class trainer:
            rollouts = ROLLOUTS
            n_epochs = 2
            timesteps = rollouts * n_epochs
            disable_progressbar = False
            headless = True

        class memory:
            rollouts = ROLLOUTS

    class pid:
        class rate:
            kp = 0.5
            ki = 0.0
            kd = 0.1
    
    class controller:
        controller_logic = False

    class log_reward:
        log_reward = True
        log_reward_interval = 100