echo "qdel all jobs"
qstat | cut -d. -f1 | sed "s;   \(.*\) 0;qdel \1;" | bash