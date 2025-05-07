//Make sure you have separate folders containing 1. your roi files, 2. your original lsm files and 
//3. an empty file for your finished images
inputDir1 = getDirectory("Choose a Directory for roi 488");
inputDir2 = getDirectory("Choose a Directory for roi 555");
inputDir3 = getDirectory("Choose a Directory for roi dpi");
outputDir = getDirectory("Choose an output Directory");
list1 = getFileList(inputDir1);
list2 = getFileList(inputDir2);
list3 = getFileList(inputDir3);


if (list1.length == list2.length)
	for (i = 0; i <list1.length; i++) {
		showProgress(i, list1.length);
		file1=list1[i];
		open(inputDir1+file1);
		id1= getTitle();
		StrInd1 = indexOf(id1, ".roi");
    	saveName1 = substring(id1, 0 , StrInd1);
    	selectWindow(id1);
    	run("ROI Manager...");
    	roiManager("Add");
        file2=list2[i];
        open(inputDir2+file2);
		id2= getTitle();
		StrInd2 = indexOf(id2, ".roi");
    	saveName2 = substring(id2, 0 , StrInd2);
    	selectWindow(id2);
    	roiManager("Add");
		file3=list3[i];
		open(inputDir3+file3);
		id3= getTitle();
		StrInd3 = indexOf(id3, ".tif");
		saveName = substring(id3, 0 , StrInd3);
		selectWindow(id3);
		run("Blue");
		run("RGB Color");
		run("Duplicate...", " ");
		selectWindow(id3);
		roiManager("Select", 0);
		roiManager("Set Color", "green");
		roiManager("Set Line Width", 0);
		run("Flatten");
		selectWindow(saveName + "-" + "2" + ".tif");
		run("Split Channels");
		selectWindow(saveName + "-" + "2" + ".tif (red)");
		//run("Properties... ", "name=" + id1 + " position=none group=none stroke=green point=Dot size=Small");
        roiManager("Select", 1);
		roiManager("Set Color", "red");
		roiManager("Set Line Width", 0);
		run("Flatten");
		selectWindow(saveName + "-" + "2" + ".tif (green)");
		run("Green");
		run("RGB Color");
		R= saveName + "-" + "2" + ".tif (red)-1";
		G= saveName + "-" + "2" + ".tif (green)";
		B= saveName + "-" + "1" + ".tif";
		run("Merge Channels...", "c1=["+R+"] c2=["+G+"] c3="+B+" create");
		selectWindow("Composite");
   		run("RGB Color");
		run("Scale...", "x=- y=- width=1140 height=800 interpolation=Bilinear average create");
		 //run("Scale...", "x=- y=- width=1320 height=800 interpolation=Bilinear average create");
		roiManager("Deselect");
        roiManager("Delete");
        saveAs("Tiff", outputDir + "/" +saveName);
		//run("Size...", "width=orgWidth height=orgHeight depth=1 average interpolation=Bilinear");
		//selectWindow(id2);
		//run("Z Project...", "projection=[Max Intensity]");
		//selectWindow("MAX"+"_"+id2);
		run("Close All");
}
else 
	print("The two input folders don't contain the same amount of images");


print("FINISHED");
