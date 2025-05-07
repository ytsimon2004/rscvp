// do the z-projection and change colors for the czi dataformat
// use for sequential gbr channels data.
// projection {'Average Intensity', 'Max Intensity', 'Min Intensity', 'Standard Deviation', 'Median'}
// TODO foreach version

function zprojLookUp(fname, color) {
	selectImage(fname);
	print("PROCESS:" + fname + ">> "+ color);
	run("Z Project...", "projection=[Median]");
	selectImage(fname);
	close();
	run(color);
	run("RGB Color");
}


function addScalebar(image_list) {
	
//	factor = getPixel2LenFactor() TBC .czi parsing error?
	
    for (i=0; i<image_list.length; i++) {
        image = image_list[i];
        selectImage(image);
        run("Scale Bar...", "width=1000 height=500 thickness=12 bold hide overlay");
    }
}

function getPixel2LenFactor() {
	// return floating type of pixel to actual length factor 
    info = getImageInfo();
    index1 = indexOf(info, "X Resolution");
    if (index1==-1)
    return "";
    index1 = indexOf(info, ":", index1);

    if (index1==-1)
    return "";
    index2 = indexOf(info, "\n", index1);

    value = substring(info, index1+1, index2);
    pixel_index = indexOf(value, "pixels");

    return parseFloat((substring(value, 0, pixel_index)))
}

// ### //

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


zprojLookUp(green_image, "Green");
zprojLookUp(blue_image, "Blue");
zprojLookUp(red_image, "Red");

zproj_image_list = getList("image.titles");

//addScalebar(zproj_image_list)



