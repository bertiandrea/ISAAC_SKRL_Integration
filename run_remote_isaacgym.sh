#!/bin/bash

# Verifica che un numero di display sia stato fornito
if [ -z "$1" ]; then
    echo "Usage: $0 <display number> [conda_env_name]"
    exit 1
fi

DISPLAY_NUM=$1
CONDA_ENV=${2:-rlgpu}  # Nome dell'ambiente Conda, predefinito a rlgpu

export DISPLAY=:$DISPLAY_NUM

echo "Using DISPLAY=$DISPLAY"
echo "Using Conda environment: $CONDA_ENV"

SCREEN_RES="1920x1080x24"

# Verifica se il display è già in uso
if [ -e /tmp/.X$DISPLAY_NUM-lock ]; then
    echo "Display :$DISPLAY_NUM is already in use!"
    exit 1
fi

# Funzione cleanup al termine dello script
cleanup() {
    echo "Stopping Xvfb and cleaning up..."
    kill $XVFB_PID 2>/dev/null
    pkill -f "gnome-session"
    pkill -f "x11vnc -display :$DISPLAY_NUM"
}
trap cleanup EXIT

# Avvia Xvfb
Xvfb $DISPLAY -screen 0 $SCREEN_RES &
XVFB_PID=$!

sleep 2

# Avvia sessione D-Bus e GNOME
eval $(dbus-launch --sh-syntax)
export DBUS_SESSION_BUS_ADDRESS
export GNOME_SHELL_SESSION_MODE=ubuntu
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=GNOME
export GDMSESSION=gnome

gnome-session --session=gnome &

sleep 5

# Salva variabili d’ambiente per sessioni future
echo "export DISPLAY=$DISPLAY" > /tmp/gnome_vnc_env.sh
echo "export DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS" >> /tmp/gnome_vnc_env.sh
chmod +x /tmp/gnome_vnc_env.sh

# Avvia x11vnc sulla porta corretta
x11vnc -display $DISPLAY -nopw -forever -bg -rfbport $((5900 + DISPLAY_NUM))

# Inizializza Conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "Conda non trovato in $HOME/miniconda3. Verifica l'installazione."
    exit 1
fi

# Avvia il training
python -m satellite.train

# Mantieni lo script attivo fino alla chiusura di Xvfb
wait $XVFB_PID
