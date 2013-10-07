# calls plot_model_performances with log folder and log file automatically set to the current time
DATE=`date "+%F-%T"`
echo $DATE
python plot_model_performances.py $1 $2 ../logs/process_logs/$DATE &> ../logs/parent_logs/$DATE
echo $PYTHONPATH