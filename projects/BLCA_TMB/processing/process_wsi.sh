for i in {0..1};do
    nohup python process_wsi.py --start ${i} --chunksize 192 >& ${i}_mask.out & 
done