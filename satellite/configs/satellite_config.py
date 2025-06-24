# satellite_config.py

from satellite.configs.base_config import BaseConfig

from pathlib import Path
import numpy as np

NUM_ENVS = 1024
N_EPOCHS = 4096
HEADLESS = True
FORCE_RENDER = False
PROFILE = False
DEBUG_ARROWS = False
HEARTBEAT = False

ROLLOUTS = 8

class SatelliteConfig(BaseConfig):
    set_seed = False
    seed = 42

    physics_engine = 'physx'

    rl_device="cuda:0"
    sim_device="cuda:0"
    graphics_device_id=0
    headless=HEADLESS
    virtual_screen_capture=False
    force_render=FORCE_RENDER

    profile = PROFILE
    
    heartbeat = HEARTBEAT
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

        debug_arrows = DEBUG_ARROWS
        
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
            
            discount_factor = 0.99 #(γ) Future reward discount; balances immediate versus long-term return.
            learning_rate = 1e-3 #Step size for optimizer (e.g. Adam) when updating policy and value networks.
            grad_norm_clip = 0.5 #Maximum norm value to clip gradients, preventing exploding gradients.
            ratio_clip = 0.2 #(ϵ) PPO’s clipping threshold on the policy probability ratio to constrain updates.
            value_clip = 0.2 #Clipping range for value function targets to stabilize value updates.
            clip_predicted_values = False #If enabled, clips the new value predictions to lie within the range defined by value_clip around the old predictions.
            entropy_loss_scale = 0.01 #Coefficient multiplying the entropy bonus; encourages exploration when > 0.
            value_loss_scale = 1.0 #Coefficient weighting the value function loss in the total loss.
            kl_threshold = 0 #Optional early-stop threshold on KL divergence between old and new policies (0 disables).
            random_timesteps = 0 #Number of initial timesteps with random actions before learning or policy-driven sampling.
            learning_starts = 0 #Number of environment steps to collect before beginning any gradient updates.
            
            class experiment:
                    write_interval = "auto"
                    checkpoint_interval = "auto"
                    directory = "./runs/satellite"
                    wandb = False

        class trainer:
            rollouts = ROLLOUTS
            n_epochs = N_EPOCHS
            timesteps = rollouts * n_epochs
            disable_progressbar = False
            headless = HEADLESS

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