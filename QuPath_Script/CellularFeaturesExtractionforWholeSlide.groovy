setImageType('BRIGHTFIELD_H_E');
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.color.ColorDeconvolutionStains;
import qupath.lib.analysis.algorithms.EstimateStainVectors;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
int MAX_PIXELS = 4000*4000;

if (!getCurrentServer().getPath().contains("--series, 0")){
 print("Working on None macro .scn file: "+getCurrentServer().getPath())
for (int i = 0; i < 2; i++) {
def viewer = getCurrentViewer()
def imageData = viewer.getImageData()
ColorDeconvolutionStains stains = imageData.getColorDeconvolutionStains();


PathObject pathObject = imageData.getHierarchy().getSelectionModel().getSelectedObject();
ROI roi = pathObject == null ? null : pathObject.getROI();
		if (roi == null){
			roi = ROIs.createRectangleROI(0, 0, imageData.getServer().getWidth(), imageData.getServer().getHeight(), ImagePlane.getDefaultPlane());
			}
double downsample = Math.max(1, Math.sqrt((roi.getBoundsWidth() * roi.getBoundsHeight()) / MAX_PIXELS));
RegionRequest request = RegionRequest.createInstance(imageData.getServerPath(), downsample, roi);
img = imageData.getServer().readBufferedImage(request);
img = EstimateStainVectors.smoothImage(img);
int[] rgb = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());
int[] rgbMode = EstimateStainVectors.getModeRGB(rgb);
String rMax = rgbMode[0];
String gMax = rgbMode[1];
String bMax = rgbMode[2];


ParameterList params = new ParameterList()
				.addDoubleParameter("minStainOD", "Min channel OD", 0.05, "", "Minimum staining OD - pixels with a lower OD in any channel (RGB) are ignored (default = 0.05)")
				.addDoubleParameter("maxStainOD", "Max total OD", 1.0, "", "Maximum staining OD - more densely stained pixels are ignored (default = 1)")
				.addDoubleParameter("ignorePercentage", "Ignore extrema", 1.0, "%", "Percentage of extreme pixels to ignore, to improve robustness in the presence of noise/other artefacts (default = 1)")
				.addBooleanParameter("checkColors", "Exclude unrecognised colors (H&E only)", false, "Exclude unexpected colors (e.g. green) that are likely to be caused by artefacts and not true staining");
//				.addDoubleParameter("ignorePercentage", "Ignore extrema", 1., "%", 0, 20, "Percentage of extreme pixels to ignore, to improve robustness in the presence of noise/other artefacts");

double minOD = params.getDoubleParameterValue("minStainOD");
double maxOD = params.getDoubleParameterValue("maxStainOD");
double ignore = params.getDoubleParameterValue("ignorePercentage");
boolean checkColors = true; // Only accept if H&E
ignore = Math.max(0, Math.min(ignore, 100));
				
	
ColorDeconvolutionStains stainsNew = EstimateStainVectors.estimateStains(img, stains, minOD, maxOD, ignore, checkColors);


hematoxylin_r = stainsNew.getStain(1).getRed().toString()
hematoxylin_g = stainsNew.getStain(1).getGreen().toString()
hematoxylin_b = stainsNew.getStain(1).getBlue().toString()
eosin_r = stainsNew.getStain(2).getRed().toString()
eosin_g = stainsNew.getStain(2).getGreen().toString()
eosin_b = stainsNew.getStain(2).getBlue().toString()
residual_r = stainsNew.getStain(3).getRed().toString()
residual_g = stainsNew.getStain(3).getGreen().toString()
residual_b = stainsNew.getStain(3).getBlue().toString()

hematoxylin_rgb_string = hematoxylin_r + " " + hematoxylin_g + " " + hematoxylin_b
eosin_rgb_string = eosin_r + " " + eosin_g + " " + eosin_b
residual_rgb_string = residual_r + " " + residual_g + " " + residual_b
background_rgb = rMax + " " + gMax + " " + bMax
final_input_string = '{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : ' + "\"" + hematoxylin_rgb_string + "\"" + ', "Stain 2" : "Eosin", "Values 2" : ' + "\"" + eosin_rgb_string + "\"" + ', "Background" : ' + 
"\" " + background_rgb + " \"}"

setColorDeconvolutionStains(final_input_string)
}//loop-2-times bracket

//----------End of the auto and Exclude unrecognised colors---------------
runPlugin('qupath.imagej.detect.tissue.SimpleTissueDetection2', '{"threshold": 200,  "requestedPixelSizeMicrons": 20.0,  "minAreaMicrons": 10000.0,  "maxHoleAreaMicrons": 1000000.0,  "darkBackground": false,  "smoothImage": true,  "medianCleanup": true,  "dilateBoundaries": false,  "smoothCoordinates": true,  "excludeOnBoundary": false,  "singleAnnotation": true}');
selectAnnotations();
runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.3,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 60.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 3.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true,  "thresholdCompartment": "Nucleus: Eosin OD mean",  "thresholdPositive1": 0.2,  "thresholdPositive2": 0.4,  "thresholdPositive3": 0.6000000000000001,  "singleThreshold": true}');
//--------------------End of detection and positive count--------------




def reg = ~/^file:/;
print "reg"
print reg
def out_path = getCurrentServer().getPath()
print out_path
int length  = out_path.length()

slash_pointer = out_path.lastIndexOf('/');
out_path = out_path.substring(slash_pointer+1, out_path.length());
print out_path


save_path = "Feature/"  //CHANGE sve path here
save_path = save_path + out_path + ".txt"
saveDetectionMeasurements(save_path)
//-----------------End of saving file------------

}
else{
print("Skipping macro .scn file: "+getCurrentServer().getPath())

}
