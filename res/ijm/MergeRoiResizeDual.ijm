// for single image resize roi-containing tif (Dual RG channels)
// Note should import .roi as gr sequentially
// TODO foreach version
cur_img=getTitle();
selectImage(cur_img);
run("RGB Color");


roiManager("Select", 0);
roiManager("Rename", "g");
RoiManager.setGroup(0);
RoiManager.setPosition(0);
roiManager("Set Color", "green");
roiManager("Set Line Width", 0);


roiManager("Select", 1);
roiManager("Rename", "r");
RoiManager.setGroup(0);
RoiManager.setPosition(1);
roiManager("Set Color", "red");
roiManager("Set Line Width", 0);

roiManager("Deselect");
roiManager("Show All");
run("Flatten");
run("Scale...", "x=- y=- width=1140 height=800 interpolation=Bilinear average create");
