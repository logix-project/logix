for i in 0.000001 0.000002 0.000003 0.000004 0.000005 0.000006 0.000007 .000008 .000009 .00001 0.00005 0.0001
do
    python compute_influences_pca.py --threshold $i --lora
done
wait
echo "All done"