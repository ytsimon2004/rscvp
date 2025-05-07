inputDir = getDirectory("/media/areca/LabPapers/SCOtherInputs/Data/Data_analyzing_Firdaouss");

fileList1 = getFileList(inputDir); 

setBatchMode(true); 
for (i = 0; i < fileList1.length; i++) { 
   showProgress(i, fileList1.length); 
   file1 = fileList1[i]; 
   open(inputDir+file1); 
   id1 = getTitle();  
   run("Z Project...", "projection=[Max Intensity]");
   selectWindow(id1); 
   close;
   selectWindow("MAX"+"_"+id1); 
   run("Split Channels"); 
   chnumber= getList("image.titles"); 
   
   for (c = 0; c < chnumber.length; c++) {
   	cn=c+1; 
   	selectWindow("C"+cn+"-"+"MAX"+"_"+id1); 
   	getLut(reds, greens, blues); 
   	
   	if (blues[255] == 255) { 
   		idch1 = getTitle();} 
   	else if(greens[255] == 255) {
   		idch2 = getTitle();}
   	}
   	//run("Merge Channels...","c1=&idch1 c2=&idch2 c3=&idch3 create");
   	run("Merge Channels...","c1=&idch1 c2=&idch2 create");
   	run("RGB Color");
   	selectWindow("MAX"+"_"+id1 + " (RGB)");

   	StrInd1 = indexOf(id1, ".lsm");
    saveName= substring(id1,0, StrInd1);
   	saveName1 = saveName+".tif";
   	//showMessage(saveName1);
   	
   	saveAs("Tiff", inputDir + "/" + saveName);
   	run("Close All"); 
}
   	showMessage("Finished splitting channels!"); 