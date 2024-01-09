#!/bin/sh

function time(){
	echo `date +%s`	
}

function diff_time(){
	TIME_A=$1
	TIME_B=$2
	PT=`expr ${TIME_B} - ${TIME_A}`
	H=`expr ${PT} / 3600`
	PT=`expr ${PT} % 3600`
	M=`expr ${PT} / 60`
	S=`expr ${PT} % 60`
	echo "${H}:${M}:${S}"
}

JOBS=$(ls ./abci/abci.job.run.*)
echo $JOBS
# groupid=gab50194
groupid=gaa50004
arid=1212

N=1
if [ "$#" -gt 0 ]; then
	N=$1
fi

echo repeat count : N=$N

function has_jobs(){
        s=$(qstat | grep $(whoami))
        if [ -z "$s" ] ; then
                echo "1"
        else
                echo "0"
        fi
}
function wait_jobs(){
        count=4
        sleep 1
        while [ $(has_jobs) = 0 ]; do
                printf "\rwait ...$count s"
                sleep 4
                count=$((count+4))
        done
        printf "\n"
}


#echo has_jobs=$(has_jobs)
for ((i=1; i<=$N; i++)); do
        echo ####### $i ######
        for S in $JOBS; do
                # echo job_name=$S
                # rm -rf ~/dump/* && qsub -g $groupid $S
                # rm -rf ~/dump/* && qsub -ar $arid -g $groupid $S
                #cmd="qsub -ar $arid -g $groupid $S"
                # cmd="rm -rf ~/dump/* && qsub -g $groupid $S"
                cmd="qsub -g $groupid $S"
		echo CMD_[$i/$N]=$cmd
		eval $cmd
                wait_jobs
        done
done

