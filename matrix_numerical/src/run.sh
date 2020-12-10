for i in 4; do
    python selfplay_train.py > console_$i.txt &
    sleep 10
done
