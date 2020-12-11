for i in $(seq 0 3); do
    python selfplay_train_asmp.py > console_asmp_$i.txt &
    sleep 10
done