for i in {1..12};do
    nohup python extract_patches.py -n 12 -i ${i}  >& ${i}_EP.out & 
done
