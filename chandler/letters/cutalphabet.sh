for i in `seq 0 25`;
do
    offset=$(($i*10))
    convert alphabet8bit.tif -crop 10x10+$offset $i.tif
done
