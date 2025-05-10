# satellite_config.py

from satellite.configs.base_config import BaseConfig
from pathlib import Path

import isaacgym #BugFix
import torch

class SatelliteConfig(BaseConfig):
    seed = 42
    physics_engine = 'physx'

    class env:
        epoch_length = 16
        # lunghezza del rollout per ciascun ambiente: numero di passi di simulazione che ciascun env
        # compie prima di aggiornare i pesi dell’agente (qui 2048 passi totali divisi equamente tra gli env)

        n_mini_epochs = 8
        # passate di ottimizzazione (SGD) per aggiornamento: quante volte il PPO
        # riesamina e riutilizza i dati raccolti durante il rollout per affinare i gradienti

        minibatch_size = 1024
        # dimensione del minibatch: numero di transizioni campionate casualmente dai dati del rollout
        # usate in ogni singolo passo di calcolo del gradiente

        n_epochs = 8192
        # numero totale di aggiornamenti dell’agente: quante volte (cicli) si esegue il
        # processo di rollout + ottimizzazione lungo l’intero training

        num_envs = 4096
        # numero di ambienti paralleli: quante istanze indipendenti dell’ambiente vengono eseguite simultaneamente 
        # per raccogliere dati in parallelo e aumentare l’efficienza
        
        num_observations = 11 # [x,y,z,w, dx,dy,dz,dw, ax,ay,az]

        num_states = 14 # [x,y,z,w, dx,dy,dz,dw, vx,vy,vz, ax,ay,az]

        num_actions = 3
        
        env_spacing = [4.0, 4.0, 4.0]

        sensor_noise_std = 0.0
        actuation_noise_std = 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        threshold_ang_goal = 0.01745        # soglia in radianti per orientamento
        threshold_vel_goal = 0.01745        # soglia in rad/sec per la differenza di velocità
        overspeed_ang_vel = 3.141           # soglia in rad/sec per l'overspeed
        episode_length_s = 30              # soglia in secondi per la terminazione di una singola simulazione
        
        #clip_actions = 1
        #clip_observations = 1

        #badly_terminated_envs_penalty = 5.0

    class asset:
        root = str(Path(__file__).resolve().parent.parent)
        file = "satellite.urdf"
        name = "satellite"

        init_pos_p = [0, 0, 0]    # posizione iniziale del satellite [x,y,z]
        init_pos_r = [0, 0, 0, 1] # attitude iniziale del satellite [x,y,z,w]

    class sim:
        dt = 1.0 / 60.0
        gravity = [0.0, 0.0, 0.0] # [m/s^2]
        up_axis = 'z'
        use_gpu_pipeline = torch.cuda.is_available()

        class physx:
            use_gpu = torch.cuda.is_available()
