//Make sure you have separate folders containing 1. your roi files, 2. your original lsm files and 
//3. an empty file for your finished images
inputDir1 = getDirectory("Z:\LabPapers\SC_nonVisualInput\00828_M1_roi");
inputDir2 = getDirectory("Z:\LabPapers\SC_nonVisualInput\00828_M1_lsm");
outputDir = getDirectory("Choose a Directory");
list1 = getFileList(inputDir1);
list2 = getFileList(inputDir2);


if (list1.length == list2.length)
	for (i = 0; i <list1.length; i++) {
		showProgress(i, list1.length);
		file1=list1[i];
		open(inputDir1+file1);
		id1= getTitle();
		StrInd1 = indexOf(id1, ".roi");
    	saveName1= substring(id1,0, StrInd1);
		file2=list2[i];
		open(inputDir2+file2);
		id2= getTitle();
		StrInd2 = indexOf(id2, ".tif");
		saveName = substring(id2, 0 , StrInd2);
		selectWindow(id2);
		orgWidth = getWidth();
		orgHeight=getHeight();
		selectWindow(id1);
		run("ROI Manager...");
		roiManager("Add");
		roiManager("Set Color", "red");
		roiManager("Set Line Width", 0);
		selectWindow(id1);
		run("Select None");
		selectWindow(id1);
		run("Size...", "width=orgWidth height=orgHeight depth=1 average interpolation=Bilinear");
		run("From ROI Manager");
		run("Flatten");
		roiManager("Delete");
		selectWindow(saveName1 + "-" + "1" + ".roi");
		run("RGB Color");
		selectWindow(id2);
		//run("Z Project...", "projection=[Max Intensity]");
		//selectWindow("MAX"+"_"+id2);
		run("Split Channels");
   		selectWindow("C1"+ "-"+id2);
   		run("RGB Color");
   		selectWindow("C3"+ "-"+id2);
   		run("RGB Color");
   		R= saveName1 + "-" + "1" + ".roi";
   		G = "C3"  + "-"+ id2;
   		B = "C1"  + "-"+ id2; 
   		run("Merge Channels...","c1="+R+" c2="+G+" c3="+B+" create");
   		selectWindow("Composite");
   		run("RGB Color");
   		//uncomment one of the next lines based on which plane you are analyzing. First is Sagittal, second is coronal
   		run("Scale...", "x=- y=- width=1320 height=800 interpolation=Bilinear average create");
		//run("Scale...", "x=- y=- width=1140 height=800 interpolation=Bilinear average create");
		saveAs("Tiff", outputDir + "/" +saveName);
		run("Close All");
}
else 
	print("The two input folders don't contain the same amount of images");


print("FINISHED");