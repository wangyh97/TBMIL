for i in {1..6};do
    nohup python extract_patches.py -n 64 -i ${i}  >& ${i}_EP.out & 
done
