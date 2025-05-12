Start-Process wt -ArgumentList '-w 0 nt --title "TensorBoard" -p "PowerShell" powershell -NoExit -Command "ssh -tt -J betajump andreaberti@192.168.1.3 ''/home/andreaberti/miniconda3/bin/tensorboard --logdir ./ISAAC_SKRL_Integration/runs --port 6006''"'

Start-Process wt -ArgumentList '-w 0 nt --title "Tunnel 5999" -p "PowerShell" powershell -NoExit -Command "ssh -tt -L 59000:localhost:5999 -C -N -J betajump andreaberti@192.168.1.3"'

Start-Process wt -ArgumentList '-w 0 nt --title "Tunnel 6006" -p "PowerShell" powershell -NoExit -Command "ssh -tt -L 59001:localhost:6006 -C -N -J betajump andreaberti@192.168.1.3"'

Start-Process wt -ArgumentList '-w 0 nt --title "SSH Shell" -p "PowerShell" powershell -NoExit -Command "ssh -tt -J betajump andreaberti@192.168.1.3"'

Start-Process wt -ArgumentList '-w 0 nt --title "NVIDIA" -p "PowerShell" powershell -NoExit -Command "ssh -tt -J betajump andreaberti@192.168.1.3 ''watch nvidia-smi''"'
