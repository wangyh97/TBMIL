# nohup python feature_extraction.py --scale 10 --extractor cTransPath --gpu 0 > fe_1.out & #跑完了
# nohup python feature_extraction.py --scale 20 --extractor cTransPath --gpu 1 > fe_2.out & #running


# nohup python feature_extraction.py --scale 10 --extractor saved --load TCGA_high --outdim 256 --gpu 0 > fe_5.out &
# nohup python feature_extraction.py --scale 20 --extractor saved --load TCGA_high --outdim 256 --gpu 0 > fe_6.out &
# nohup python feature_extraction.py --scale 10 --extractor res --layers 18 --gpu 0 > fe_7.out & 跑完了
# nohup python feature_extraction.py --scale 20 --extractor res --layers 18 --gpu 1 > fe_8.out &
nohup python feature_extraction.py --scale 10 --extractor res --layers 50 --gpu 0 > fe_9.out & #running
# nohup python feature_extraction.py --scale 20 --extractor res --layers 50 --gpu 1 > fe_10.out &

nohup python feature_extraction.py --scale 10 --extractor retccl --gpu 0 > fe_3.out & # 跑完了
# nohup python feature_extraction.py --scale 20 --extractor retccl --gpu 1 > fe_4.out & #跑完了