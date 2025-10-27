nohup python feature_extraction.py --scale 10 --extractor pretrained_resnet18 &> fe_10_pretrained_resnet18.out 
# nohup python feature_extraction.py --scale 10 --extractor pretrained_resnet50 &> fe_10_pretrained_resnet50.out 
# nohup python feature_extraction.py --scale 10 --extractor retccl --outdim 2048 &> fe_10_retccl.out 
# nohup python feature_extraction.py --scale 10 --extractor cTransPath --outdim 768 &> fe_10_cTransPath.out 
nohup python feature_extraction.py --scale 10 --extractor c16_high --outdim 256 &> fe_10_cTransPath.out

# nohup python feature_extraction.py --scale 20 --extractor pretrained_resnet18 &> fe_20_pretrained_resnet18.out 
# nohup python feature_extraction.py --scale 20 --extractor pretrained_resnet50 &> fe_20_pretrained_resnet50.out 
# nohup python feature_extraction.py --scale 20 --extractor retccl --outdim 2048 &> fe_20_retccl.out 
# nohup python feature_extraction.py --scale 20 --extractor cTransPath --outdim 768 &> fe_20_cTransPath.out 

