// do the z-projection and change colors for the lsm dataformat
// use for sequential brg channels data.
// projection {'Average Intensity', 'Max Intensity', 'Min Intensity', 'Standard Deviation', 'Median'}

function zprojLookUp(fname, color) {
	selectImage(fname);
	print("PROCESS:" + fname + ">> "+ color);
	run("Z Project...", "projection=[Max Intensity]");
	selectImage(fname);
	close();
	run(color);
	run("RGB Color");
}

cur_img=getTitle();
selectImage(cur_img);
run("Split Channels");

imageList = getList("image.titles");
if (imageList.length != 3)
	exit("chanel less than 3");

// assume image are in a correct order
green_image = imageList[0];
blue_image = imageList[1];
red_image = imageList[2];


zprojLookUp(green_image, "Blue");
zprojLookUp(blue_image, "Red");
zprojLookUp(red_image, "Green");

zproj_image_list = getList("image.titles")