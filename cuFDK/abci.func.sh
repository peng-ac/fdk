
function show-strong2k(){
	clear; for ((i=4; i<=2048; i*=2)); do  echo "------- $i ----------";  cat -n *.t2 | grep " 02048-02048-02048, $(printf "%05d" $i)"; done;
}

function show-strong4k(){
	clear; for ((i=32; i<=2048; i*=2)); do  echo "------- $i ----------";  cat -n *.t2 | grep " 04096-04096-04096, $(printf "%05d" $i)"; done;
}

function show-weak2k(){
	 clear; for ((i=4; i<=2048; i*=2)); do  echo "------- $i ----------";  cat -n *.t2 | grep "02048-02048-02048, $(printf "%05d" $i)"; done;
}

function show-weak4k(){
	 clear; for ((i=32; i<=2048; i*=2)); do  echo "------- $i ----------";  cat -n *.t2 | grep "04096-04096-04096, $(printf "%05d" $i)"; done;
}

function abci-backup(){
	if [ $# -ne 1 ]; then
		echo "invalid argument count"
		exit 1
	fi
	DST=$1
	echo "backup dir : " $DST

	mkdir -p $DST

	mv *.t1 $DST
	mv *.t2 $DST
	mv abci.job.run.sh.* $DST
}




