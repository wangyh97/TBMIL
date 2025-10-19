# echo "start!"

# # 使用数组保存所有进程的 pid
# pids=()
# for i in {1..14}; do
#    nohup python reduce.py --scale 10 --extractor res --layers 18 --chunksize 26 --cpu_index ${i} &> ./reduce/reduce_10X_res18_8_${i}.out &
#    pids+=("$!")
# done

# for pid in "${pids[@]}"; do
#   wait "$pid" 
# done
# echo "reduce_10X_res18_8 done"


# 同样的方式执行其他两个循环

# 9-17 rerun
pids=()
for i in {1..14}; do
   nohup python reduce.py --scale 10 --extractor res --layers 50 --chunksize 26 --cpu_index ${i} &> ./reduce/reduce_10X_res50_8_${i}.out &  
   pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done 
echo "reduce_10X_res50_8 done"

# pids=( )

# for i in {1..14}; do
#   nohup python reduce.py --scale 10 --extractor retccl --chunksize 26 --cpu_index ${i} &> ./reduce/reduce_10X_retccl_8_${i}.out &
#   pids+=("$!") 
# done

# # 等待所有进程完成
# for pid in "${pids[@]}"; do
#   wait "$pid"
# done
# echo "reduce_10X_retccl_8 done"