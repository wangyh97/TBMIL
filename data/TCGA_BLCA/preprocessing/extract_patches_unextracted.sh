for i in {1..6};do
    nohup python extract_patches_unextracted.py -n 11 -i ${i}  >& ${i}_EP.out & 
done
