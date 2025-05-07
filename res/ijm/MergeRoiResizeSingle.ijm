// for single image resize roi-containing tif (Single G channels)
//

cur_img=getTitle();
selectImage(cur_img);
run("RGB Color");


roiManager("Select", 0);
roiManager("Rename", "r");
RoiManager.setGroup(0);
RoiManager.setPosition(0);
roiManager("Set Color", "red");
roiManager("Set Line Width", 0);


roiManager("Deselect");
roiManager("Show All");
run("Flatten");
run("Scale...", "x=- y=- width=1140 height=800 interpolation=Bilinear average create");
