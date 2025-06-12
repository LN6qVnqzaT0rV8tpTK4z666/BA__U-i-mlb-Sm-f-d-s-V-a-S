#!/bin/bash
# BA__Projekt/scripts/service__restart_all.sh

# Find the PID of the TensorBoard process
PID=$(ps aux | grep '[t]ensorboard' | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "Killing TensorBoard process with PID: $PID"
    kill "$PID"
else
    echo "No TensorBoard process running. Nothing to kill."
fi