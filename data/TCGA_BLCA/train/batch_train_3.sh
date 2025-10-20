cd training_details/10X_5fold_TCGA_high

# cd ../10X_5fold_c16_high_NoSched
# nohup bash -c "cat 10X_5fold_c16_high_NoSched_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_NoSched finished"

# cd ../10X_5fold_c16_high_warmup
# nohup bash -c "cat 10X_5fold_c16_high_warmup_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_warmup finished"

# cd ../10X_5fold_c16_high_reg4
# nohup bash -c "cat 10X_5fold_c16_high_reg4_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_reg4 finished"

# cd ../10X_5fold_c16_high_reg3
# nohup bash -c "cat 10X_5fold_c16_high_reg3_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_reg3 finished"

# cd ../10X_5fold_c16_high_reg2
# nohup bash -c "cat 10X_5fold_c16_high_reg2_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_reg2 finished"

# cd ../10X_5fold_c16_high_reg1
# nohup bash -c "cat 10X_5fold_c16_high_reg1_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_reg1 finished"

# cd ../10X_5fold_c16_high_dropout_node
# nohup bash -c "cat 10X_5fold_c16_high_dropout_node_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_dropout_node finished"

# cd ../10X_5fold_c16_high_dropout_node
# nohup bash -c "cat 10X_5fold_c16_high_dropout_node_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_dropout_node finished"

# cd ../10X_5fold_c16_high_finetune_1
# nohup bash -c "cat 10X_5fold_c16_high_finetune_1_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_finetune_1 finished"

# cd ../10X_5fold_c16_high_finetune_2
# nohup bash -c "cat 10X_5fold_c16_high_finetune_2_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_finetune_2 finished"

# cd ../10X_5fold_c16_high_finetune_3
# nohup bash -c "cat 10X_5fold_c16_high_finetune_3_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
# echo "10X_5fold_c16_high_finetune_3 finished"

cd ../10X_5fold_c16_high_finetune_4
nohup bash -c "cat 10X_5fold_c16_high_finetune_4_3.sh | xargs -P 12 -I {} bash -c '{} && echo \"command finished\"'" &>> output.log
echo "10X_5fold_c16_high_finetune_4 finished"