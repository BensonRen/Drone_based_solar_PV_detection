#!/bin/bash

#export PYTHONPATH="/home/sr365/Gaia/mrs/:$PYTHONPATH"
export PYTHONPATH="/scratch/sr365/mrs/:$PYTHONPATH"

# Waiting orders
#PID=6248
#while [ -e /proc/$PID ]
#do
#    echo "Process: $PID is still running" 
#        sleep 1m
#done

TIME=`date`
PWD=`pwd`
# The command to execute
#COMMAND=hyper_sweep.py
#COMMAND=compare.py
#COMMAND=object_pr.py
#COMMAND=cut_RTI.py
#COMMAND=aggregate_pr_curves.py
COMMAND=infer_catalyst.py
#COMMAND=change_to_sat_res.py
#COMMAND=train.py
#COMMAND="train.py --config config_0629_Rwanda.json"
#COMMAND="train.py --config config_ben_0406_h3_RTI.json"
#COMMAND="train.py --config config_ben_0407_h2RTI_mixed.json"
#COMMAND="train.py --config config_ben_0407_h3RTI_mixed.json"
#COMMAND="train.py --config config_ben_sat_res_h2.json"
SPACE='        '
SECONDS=0

nohup python $COMMAND 1>output.out 2>error.err & 
#nohup python $COMMAND 1>outputcutRTI.out 2>errorcurRTI.err & 
#nohup python $COMMAND 1>output02.out 2>error02.err & 
echo $! > pidfile.txt

# Make sure it waits 10s for super fast programs
sleep 10s

PID=`cat pidfile.txt`
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
        sleep 3m
done
#If the running time is less than 200 seconds (check every 180s), it must have been an error, abort
duration=$SECONDS
limit=10
if (( $duration < $limit )) 
then
    echo The program ends very shortly after its launch, probably it failed
    exit
fi

H=$(( $duration/3600 ))
M=$((( ($duration%3600 )) / 60 ))
S=$(( $duration%60 ))
#echo $H
#echo $M
#echo $S

CURRENTTIME=`date`
{
	echo To: rensimiao.ben@gmail.com
	echo From: Cerus Machine
	echo Subject: Your Job has finished!
	echo -e "Dear mighty Machine Learning researcher Ben, \n \n"
	echo -e  "    Your job has been finished and again, you saved so many fairies!!!\n \n"
	echo -e  "Details of your job:\n
        Job:  $COMMAND \n   
	PID:   `cat pidfile.txt` \n 
	TIME SPENT: $H hours $M minutes and $S seconds \n
        StartTime:   $TIME \n 
        ENDTIME: $CURRENTTIME \n
	PWD:  $PWD\n"
} | ssmtp rensimiao.ben@gmail.com

echo "Process $PID has finished"

#Copying the parameters to the models folder as a record
#Lastfile=`ls -t models/ | head -1`
#mv parameters.txt models/$Lastfile/.
#cp parameters.py models/$Lastfile/.
#cp running.log models/$Lastfile/.
#cp running.err models/$Lastfile/.
