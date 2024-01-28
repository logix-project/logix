# for i in 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001
# do
#     python compute_influences_scheduler.py --threshold $i --lora
# done
# wait
# echo "All done"
for i in 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 0.002 0.003 0.004 0.005 0.01, 0.03, 0.05, 0.07, 0.09, 0.1
do
    python compute_influences_scheduler.py --threshold $i --lora
done
wait
echo "All done"