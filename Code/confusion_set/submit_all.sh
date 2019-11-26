for f in *.sbatch
do
	echo "Starting job for: $f"
    sbatch $f
done
