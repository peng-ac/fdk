#!/usr/bin/env bash


function tm(){
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


function proc(){
	weakstrong=$1
	size=$2
	echo "weakstrong="$weakstrong
	echo "size="$size

	pushd abci
		./clear.sh 
		python abci-strong.py $weakstrong$size
	popd
	./abci.batch.sub.sh 16 
	./cp.sh $weakstrong $size
}

start_time=$(tm)

for size in 2k; do
	# for ws in strong weak; do
	# for ws in strong; do
	for ws in strong; do
		proc $ws $size
		diff_time $start_time $(tm)
	done	
done

# proc strong 2k
diff_time $start_time $(tm)






