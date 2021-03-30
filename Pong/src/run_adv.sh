for i in 0 1 2; do
    python adv_train.py --server_id $i > console_$i.txt &
    sleep 10
done
