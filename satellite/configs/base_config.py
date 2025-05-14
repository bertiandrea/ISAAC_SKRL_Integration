# base_config.py

import inspect

"""
config
├─ sim
│  ├─ use_gpu_pipeline          # bool
│  ├─ up_axis                   # "z" o "y"
│  ├─ dt                        # float
│  ├─ num_client_threads        # int   (opzionale)
│  ├─ substeps                  # int   (opzionale)
│  ├─ gravity                   # [float, float, float]
│  ├─ physics_engine            # “physx” o “flex”
│  ├─ physx                     # dict  (opzionale, se physics_engine=="physx")
│  │  └─ …                      # es. contact_collection, solver iterations, ecc.
│  └─ flex                      # dict  (opzionale, se physics_engine=="flex")
│     └─ …                      # opzioni specifiche Flex
└─ env
   ├─ numEnvs                   # int
   ├─ numAgents                 # int   (opzionale; default=1)
   ├─ numObservations           # int   (opzionale; default=0)
   ├─ numStates                 # int   (opzionale; default=0)
   ├─ numActions                # int
   ├─ controlFrequencyInv       # int   (opzionale; default=1)
   ├─ clipObservations          # float (opzionale; default=Inf)
   ├─ clipActions               # float (opzionale; default=Inf)
   └─ renderFPS                 # int   (opzionale; default=-1)

physx
    ├─ use_gpu                         # PhysX su GPU (bool)
    ├─ num_threads                     # Number of CPU threads for PhysX
    ├─ solver_type                     # 0=PGS, 1=TGS
    ├─ num_position_iterations         # Iterazioni di posizione (1–255)
    ├─ num_velocity_iterations         # Iterazioni di velocità (1–255)
    ├─ contact_offset                  # Distanza a cui generare i contatti
    ├─ rest_offset                     # Distanza di riposo dopo il contatto
    ├─ bounce_threshold_velocity       # Velocità soglia per rimbalzo
    ├─ max_depenetration_velocity      # Velocità massima di correzione penetrazione
    ├─ default_buffer_size_multiplier  # Scala dei buffer interni GPU
    ├─ max_gpu_contact_pairs           # Dimensione del buffer contatti GPU
    ├─ friction_offset_threshold       # Distanza soglia per vincoli di attrito
    ├─ friction_correlation_distance   # Distanza di correlazione attrito
    └─ num_subscenes                   # Numero di sotto‐scene per multithreading
"""

class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key=="__class__":
                continue
            var =  getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)
