clear
mkdir -p ~/abci-bk
mv abci.job.run.sh.* ~/abci-bk
qsub -g gaa50004 abci.job.run.sh 
