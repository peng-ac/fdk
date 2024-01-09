
N=16
for t in strong2k strong4k weak2k weak4k; do
	pushd ./abci
		./clear.sh
		python abci-strong.py $t
	popd
	# read -p "Enter key please"
	./abci.batch.sub.sh
done



