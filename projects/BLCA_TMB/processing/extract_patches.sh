# for i in {1..11};do
#     nohup python extract_patches.py -n 35 -i ${i}  >& ${i}_EP.out & 
# done

for i in {1..11};do
    nohup python extract_patches_size256.py -n 35 -i ${i}  >& ${i}_EP_256.out & 
done