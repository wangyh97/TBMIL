# nohup python feature_extraction.py --scale 10 --extractor pretrained_resnet18 --gpu 0 &> fe_10_pretrained_resnet18.out 
# nohup python feature_extraction.py --scale 10 --extractor pretrained_resnet50 --gpu 0 &> fe_10_pretrained_resnet50.out 
# nohup python feature_extraction.py --scale 10 --extractor retccl --outdim 2048 --gpu 0 &> fe_10_retccl.out 
# nohup python feature_extraction.py --scale 10 --extractor cTransPath --outdim 768 --gpu 0 &> fe_10_cTransPath.out 
nohup python feature_extraction.py --scale 10 --extractor TCGA_high --outdim 256 --gpu 0 &> fe_10_TCGA_high.out 
nohup python feature_extraction.py --scale 10 --extractor TCGA_low --outdim 256 --gpu 0 &> fe_10_TCGA_low.out 
nohup python feature_extraction.py --scale 10 --extractor c16_high --outdim 256 --gpu 0 &> fe_10_c16.out 


# nohup python feature_extraction.py --scale 20 --extractor pretrained_resnet18 --gpu 1 &> fe_20_pretrained_resnet18.out 
# nohup python feature_extraction.py --scale 20 --extractor pretrained_resnet50 --gpu 1 &> fe_20_pretrained_resnet50.out 
# nohup python feature_extraction.py --scale 20 --extractor retccl --outdim 2048 --gpu 1 &> fe_20_retccl.out 
# nohup python feature_extraction.py --scale 20 --extractor cTransPath --outdim 768 --gpu 1 &> fe_20_cTransPath.out 

