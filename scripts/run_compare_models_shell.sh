# calls run_compare_models.py with log folder and log file automatically set to the current time
DATE=`date "+%F-%T"`
echo $DATE
python run_compare_models_scripts.py $1 $2 ../logs/process_logs/$DATE &> ../logs/parent_logs/$DATE