# ISAAC_SKRL_Integration

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

Esegui lo script `run_remote_vnc.sh` specificando il numero di display (es: `1`) e l'environment (es: rlgpu):

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
