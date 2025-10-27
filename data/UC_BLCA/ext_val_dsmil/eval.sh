# 9-13
# nohup python eval.py --scale 10 --layer 50 --description 10X_5fold_res50 --fold 0,1,2,3,4 &> ./out/eval_10X_res50_fold5.out
# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_res18 --fold 0,1,2,3,4 &> ./out/eval_10X_res18_fold5.out
# nohup python eval.py --scale 20 --layer 18 --description 20X_5fold_res18 --fold 0,1,2,3,4 &> ./out/eval_20X_res18_fold5.out

# 9-14 
# command record, 分开跑的
# nohup python eval.py --scale 10 --layer 50 --gpu_index 0 --description 10X_5fold_res50_ft --fold 0,1,2,3,4 &> ./out/eval_10X_res50_fold5.out &
# nohup python eval.py --scale 10 --layer 18 --gpu_index 1 --description 10X_5fold_res18_ft --fold 0,1,2,3,4 &> ./out/eval_10X_res18_fold5.out &
# nohup python eval.py --scale 20 --load TCGA_high --feats_size 256 --gpu_index 2 --description 20X_5fold_TCGA_high_ft --fold 0,1,2,3,4 &> ./out/eval_20X_res18_fold5.out &

# 9-19

# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_res18_reg --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res18_reg.out
# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_res18_warmup --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res18_warmup.out
# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_res18_reg_warm --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res18_reg_warm.out
# nohup python eval.py --scale 10 --layer 50 --description 10X_5fold_res50_reg --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res50_reg.out
# nohup python eval.py --scale 10 --layer 50 --description 10X_5fold_res50_warmup --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res50_warmup.out
# nohup python eval.py --scale 10 --layer 50 --description 10X_5fold_res50_reg_warm --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res50_reg_warm.out

# 12-19
# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_res18_ft_dropout_node --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_res18_ft_dropout_node.out
# nohup python eval.py --scale 10 --load pretrained_resnet18 --description 10X_5fold_pretrained_resnet18_warmup_512 --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_pretrained_resnet18_warmup_512.out
# nohup python eval.py --scale 10 --load pretrained_resnet50 --description 10X_5fold_pretrained_resnet50_warmup_512 --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_pretrained_resnet50_warmup_512.out
# nohup python eval.py --scale 10 --layer 18 --description 10X_5fold_cTransPath_warmup --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_cTransPath_warmup.out

nohup python eval.py --scale 10 --extractor c16_high --description 10X_5fold_c16_high_finetune_3 --fold 0,1,2,3,4 &> ./out/eval_10X_5fold_c16_ft3.out