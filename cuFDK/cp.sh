# !/bin/bash

ROOT=~/gc

if test $# != '2'; then
	echo "arguments number error!"
	exit 1
fi

WeaKStrong=$1
Size=$2

#############
echo ParameterNumbers=$#
echo WeaKStrong=$1
echo Size=$2

mkdir -p $ROOT
mkdir -p $ROOT/$WeaKStrong
dir=$ROOT/$WeaKStrong/$Size
echo "dst_dir="$dir

mkdir -p $dir

mv abci.job.run.* $dir
mv *.t1 $dir
mv *.t2 $dir



