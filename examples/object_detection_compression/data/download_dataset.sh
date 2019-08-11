echo "Downloading Dataset..."
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting Dataset..."
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip