
function get_count(){
	s=$(qstat | wc -l) 
	#s=$(qstat | grep $(id -u -n) | wc -l) 
	return $s 
}

echo $(get_count)
