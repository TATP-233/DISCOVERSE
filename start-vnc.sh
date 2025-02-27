#!/bin/bash

# Start Xvfb
echo "Starting Xvfb..."
Xvfb ${DISPLAY} -screen 0 ${VNC_RESOLUTION}x${VNC_COL_DEPTH} &
sleep 2

# Start window manager
echo "Starting Fluxbox..."
fluxbox &
sleep 2

# Start VNC server with verbose logging and explicit port
echo "Starting x11vnc..."
x11vnc -display ${DISPLAY} -nopw -forever -verbose -rfbport ${VNC_PORT} &
sleep 2

# Start noVNC with correct port
echo "Starting noVNC..."
/usr/share/novnc/utils/launch.sh --vnc localhost:${VNC_PORT} --listen ${NO_VNC_PORT} &

while true; do
    sleep 30
done