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






