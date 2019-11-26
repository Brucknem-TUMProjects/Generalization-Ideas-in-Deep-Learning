for f in *.measures
do
	echo "Starting job for: $f"
    measures $f
done
