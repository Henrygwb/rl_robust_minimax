for i in $(seq 0 2); do
    python minimax_train.py > console_$i.txt &
    sleep 10
done