JOBS=$(ls ./abci/abci.job.run.*)
echo $JOBS

N=2

echo reoeat count : N=$N

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
	echo "$i -->" 
	for S in $JOBS; do
		echo job_name=$S
		qsub -g gaa50004 $S
		wait_jobs
	done
done
