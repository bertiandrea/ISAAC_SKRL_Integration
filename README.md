# ISAAC_SKRL_Integration

Questo progetto implementa una pipeline end-to-end per addestrare un agente di controllo dell’assetto orbitale di un satellite, sfruttando NVIDIA Isaac Gym per la simulazione fisica parallela e SKRL per l’algoritmo PPO.

---
## 1. Architettura
1. **Configuration**  
   Gestione centralizzata dei parametri di simulazione, dell’ambiente e dell’algoritmo di RL.  
2. **Environment**  
   Creazione e gestione in parallelo delle istanze di simulazione via Isaac Gym.
3. **Rewards**  
   Set modulare di funzioni di ricompensa per diverse strategie di shaping della reward.
4. **Utilities**
   Funzioni quaternion-based per campionamento random e il calcolo di differenze tra quaternioni.
6. **Models**  
   Creazione di 2 reti neurali: Policy (Gaussiana) e Value (Deterministica).
7. **Asset URDF**  
   Descrizione fisica del satellite.  
8. **Training**  
   Script di lancio che unisce parsing argomenti, istanziazione ambiente, configurazione PPO e avvio del ciclo di apprendimento.

---
## 2. Configuration (`configs/`)
- **`BaseConfig`**  
  Classe di base che individua e istanzia tutte le classi annidate come attributi.  
- **`SatelliteConfig`**  
  - **Parametri generali**:  
    - Seed, device CUDA/CPU, risoluzione schermo.
  - **Parametri `env`**:  
    - Numero di ambienti paralleli, dimensione di stato/azione/osservazione, rumore sensoriale/attuazione, soglie di terminazione (angolo, velocità, overspeed), durata epoca, ecc... 
  - **Parametri `asset`**:  
    - Percorso al file URDF, nome dell’actor, posizione e orientamento iniziale.  
  - **Parametri `sim`**:  
    - Passo di integrazione (`dt`), gravità, motore fisico (PhysX/Flex) e parametri specifici (solver, iterazioni, offset di contatto).  
  - **Parametri `rl`**:  
    - Setting dei parametri di PPO (rollout length, learning rate, clipping, scale delle perdite, checkpoint), trainer (numero di epoche, timesteps totali).

---
## 3. Environment (`envs/`)
### 3.1. `Params`
- Carica dal config tutti i parametri utili della simulazione.
### 3.2. `VecTask`  
Estende `Params` e incapsula:
- **Creazione della simulazione e degli N ambienti paralleli**.
- **Load dell’asset** negli N ambienti paralleli.
- **Viewer** per visualizzare la scena.
- Buffer Torch per osservazioni (`obs_buf`), stati (`states_buf`), reward, reset e avanzamento.
- Metodi chiave:
  - `step(actions)`: esegue una simulazione e restituisce all'algoritmo PPO `(states, reward, reset, timeout)`.
  - `reset()`: reset completo di tutti gli ambienti.
  - `render()`, `close()`, `destroy()`.

### 3.3. `SatelliteVec`  
Estende `VecTask` per:
- Estrarre da `actor_root_state_tensor` le posizioni, quaternion, velocità angolari, accelerazioni angolari popolando:
  - **`obs_buf`**: quaternione corrente, differenza quaternion, accelerazione angolare.
  - **`states_buf`**: come sopra + velocità angolari.
- Applicare la coppia di controllo sul satellite.
- Calcolare reward tramite una `RewardFunction` selezionabile.
- Verificare la terminazione per:
  - **Goal**: errore di orientamento e velocità sotto soglia.
  - **Timeout/Overspeed**: durata massima o velocità angolare eccessiva.

---
## 4. Rewards (`rewards/`)
Dall'errore di orientamento, velocità e accelerazione rispetto al goal calcola la reward di ciascun ambiente.
Implementazioni principali:
- **`TestReward`**: peso inverso agli errori di orientamento, velocità e accelerazione.  
- **`WeightedSumReward`**: somma pesata + bonus early success + penalità per saturazione.  
- **`TwoPhaseReward`**: ricompensa a due fasi, con decay esponenziale una volta sotto soglia.  
- **`ExponentialStabilizationReward`**, **`ContinuousDiscreteEffortReward`**, **`ShapingReward`**.

---
## 5. Utilities (`utils/`)
Funzioni tensor‐based per: generare N quaternioni random, calcolare la distanza angolare in radianti tra due quaternioni, calcolare la differenza tra due quaternioni.

---
## 6. Models (`models/`)
- **`Policy`**
Policy: rete a 3 layer densi, con output `mean` e `log-std` per la distribuzione gaussiana delle azioni. Utilizza `observation_space`.
- **`Value`**
Value: rete a 3 layer densi, con output singolo `valore di stato`. Utilizza `state_space` ed è quindi onniscente.

---
## 7. Asset URDF
Definizione fisica del satellite.

---
## 8. Training
 - Configurazione del simulatore, degli environment, e dell'algoritmo PPO.
 - Creazione modelli Policy/Value, agente PPO e trainer sequenziale.
 - Lancio del processo di training.

---
## 🖥️ Esecuzione Locale

1. Attiva l’ambiente Conda:

   ```bash
   conda activate rlgpu
   ```

2. Avvia il training:

   ```bash
   python -m satellite.train --reward-fn test
   ```

---

## 🌐 Esecuzione Remota con VNC

### 🔹 1. Sul Server Remoto

Esegui lo script `run_remote_vnc.sh` specificando il numero di display (es: `1`), l'environment (es: `rlgpu`) e la reward function (es: `test`):

```bash
./run_remote_vnc.sh 1 rlgpu test
```

---

### 🔹 2. Sul Computer Locale

#### a. Crea un tunnel SSH con port forwarding:

```bash
ssh -v -L 59000:localhost:$((5900 + <PORT>)) -C -N -J betajump your_user@your.remote.ip
```

Sostituisci:

* `<PORT>` con il numero usato nel comando `run_remote_vnc.sh`
* `your_user` con il tuo username SSH
* `your.remote.ip` con l’indirizzo IP del server remoto

#### b. Avvia VNC Viewer:

Connettiti a:

```
localhost:59000
```

Accederai al desktop GNOME remoto per monitorare l’allenamento.

---

## 📊 Monitoraggio con TensorBoard

### ✅ Avvio di TensorBoard sul server remoto

Assicurati che il training scriva i log in `./runs`, quindi avvia TensorBoard:

```bash
tensorboard --logdir ./runs --port 6006
```

### 🔁 Accesso remoto a TensorBoard da Windows (PowerShell)

Apri PowerShell e crea un tunnel SSH:

```powershell
ssh -L 59001:localhost:6006 -C -N -J betajump andreaberti@192.168.1.3
```

Poi apri il browser e visita:

```
http://localhost:59001
```

Vedrai la dashboard di TensorBoard in esecuzione sul server remoto.

---

## ⚙️ Installazione Miniconda & Isaac Gym

### ✅ Prerequisiti

* Sistema operativo: Linux x86\_64
* Strumenti richiesti: `wget`, `unzip` e/o `tar`

### 1️⃣ Installazione di Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

Questo installerà Miniconda in modalità batch e configurerà automaticamente la shell per l’uso di `conda`.

### 2️⃣ Download di Isaac Gym

```bash
tar -xf IsaacGym_Preview_4_Package.tar.gz
```

Estrai il pacchetto nella directory corrente.

### 3️⃣ Edit: rlgpu_conda_env.yml

```bash
 name: rlgpu
 channels:
   - pytorch
   - conda-forge
   - defaults
 dependencies:
-  - python=3.7
+  - python=3.8
   - pytorch=1.8.1
   - torchvision=0.9.1
   - cudatoolkit=11.1
   - pyyaml>=5.3.1
   - scipy>=1.5.0
   - tensorboard>=2.2.1
+  - mkl=2024.0
```

### 4️⃣ Installazione dell’Ambiente Isaac Gym

```bash
cd isaacgym
./create_conda_env_rlgpu.sh
```

Lo script creerà l’ambiente Conda `rlgpu` e installerà tutte le dipendenze necessarie.

### 5️⃣ Attivazione dell’Ambiente

```bash
conda activate rlgpu
```

## ⚙️ Installazione SKRL

```bash
conda activate rlgpu
pip install skrl["torch"]
```

## ⚙️ Opzionale per il funzionamento della GPU

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## ⚙️ Verifica dell’Installazione

```bash
conda activate rlgpu
cd isaacgym/python/examples
python joint_monkey.py
```

Se lo script si avvia correttamente, l’installazione è andata a buon fine.

## ⚙️ Opt. Installa tensorboard nell'environment base

```bash
conda activate base
pip install tensorboard
pip install standard-imghdr
```
---
