dir1 = getDirectory("Choose Source Directory");
dir2 = getDirectory("Choose Destination Directory");
list = getFileList(dir1);
setBatchMode(true);
for (i=0; i<list.length; i++) {    
	showProgress(i+1, list.length); 
	open(dir1+list[i]);
	run("Scale...", "x=0.25 y=0.25 z=1.0 width=4798 height=3363 depth=3 interpolation=Bicubic average process create");
	run("Rotate... ", "angle=180 grid=1 interpolation=Bicubic stack");
 	saveAs("TIFF", dir2+list[i]);    
 	close();
}

