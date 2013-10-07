DATE=`date "+%F-%T"`
echo $DATE
python plot_predicted_patient_curves.py $1 $2 ../logs/process_logs/$DATE &> ../logs/parent_logs/$DATE
echo $PYTHONPATH