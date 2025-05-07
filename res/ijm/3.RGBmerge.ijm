inputDir = getDirectory("/media/areca/LabPapers/SCOtherInputs/Data/Data_analyzing_Firdaouss");

fileList1 = getFileList(inputDir); 

setBatchMode(true); 
for (i = 0; i < fileList1.length; i++) { 
   showProgress(i, fileList1.length); 
   file1 = fileList1[i]; 
   open(inputDir+file1); 
   run("Rotate 90 Degrees Right");
   run("Size...", "width=1140 height=800 interpolation=Bicubic"); 
   id1 = getTitle();
   run("RGB Color"); 
   selectWindow(id1 + " (RGB)");
   StrInd1 = indexOf(id1, ".tif");
   saveName= substring(id1,0, StrInd1);
   saveName1 = saveName+".tif";
   //showMessage(saveName1);
   	
   saveAs("Tiff", inputDir + "/" + saveName);
   run("Close All"); 
}
   showMessage("Finished RGB merge!"); 