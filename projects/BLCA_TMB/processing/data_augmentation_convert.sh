#!/bin/bash

function print_verbose {
    if [ "$VERBOSE" = true ]; then
        echo "$1"
    fi
}

# 检查参数数量
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_directory> <destination_directory> <action> [verbose]"
    exit 1
fi


# 检查是否输入了"verbose"
if [ "$4" = "verbose" ]; then
    VERBOSE=true
else
    VERBOSE=false
fi

directories=("$1")
DEST_DIR="$2"
ACTION="$3"
VERBOSE="$4"

for dir_pattern in "${directories[@]}"; do
    find $dir_pattern -type d | while read -r dir; do
        echo "Processing directory: $dir"
        # 对每个目录进行处理   
        find "$dir" -type f -name 'T*.tiff' -exec sh -c '
        src="$1"   #子shell中$1即传入的文件路径      
        dir_replace="/GPUFS/sysu_jhluo_1/wangyh/data/raw_patches/"
        action="$3"
        verbose="$4"
        dest_dir="$5"
        
        relative_path="${src#"$dir_replace"}";  #向前去掉src这个变量中最短匹配的pattern，这里是$2，即SRC_DIR，以得到相对路径,因为在命令行中需要在文件夹末尾加上/，否则这里改为${src#2/}
        
        dest="$dest_dir$relative_path"; #以同样的相对路径构造dest,同样的，命令行中传入的dest末尾有/，否则这里改为<$3/$relative_path>
        mkdir -p "$(dirname "$dest")"; #返回父文件夹名并创建对应的文件夹， -p表示如果没有就创建一个
        convert "$src" $action "$dest"; #$4为action #这里$4不能用引号括起来，否则会报错 
        [ "$5" = true ] && echo "$dest done";

' _ {} "$dir" "$ACTION" "$VERBOSE" "$DEST_DIR" \; #_是占位符，表示传入sh{}的$0，即命令本身，但不需要它，后面的{}表示find传给子shell的第一个参数，即找到的文件路径。exec要求最后的反斜杠前面有空格
    done
done


# demo command: bash data_augmentation_convert.sh '/path' /path 'action' verbose,这里的第一个path一定要用单引号括起来
# bash data_augmentation_convert.sh '/GPUFS/sysu_jhluo_1/wangyh/data/raw_patches/L/169f3822-c236-42b4-b160-6d7167e17726/*/' /GPUFS/sysu_jhluo_1/wangyh/data/data_aug/ '-rotate 90' verbose