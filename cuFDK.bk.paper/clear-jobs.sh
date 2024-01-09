# qstat | cut -d. -f1 | sed "s;   \(.*\) 0;qdel \1;" | bash
qstat -u $USER | gawk '{print $1}' | xargs qdel