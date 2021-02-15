# metrics

This scripts allows:  

\* calculate precision and recall (per class)    
\* calculate collision matrix   
\* show false positive detections  

### Installation

Install the dependencies from `requierements.txt`  

### Running the evaluation

Run the following command:

`python metrics.py -g ./truth -p ./preds -n ./your_file.names`  
where:  

`-g, --gt_dir <folder>`  : folder with ground truth .txt files with bounding boxes  

`-p, --pred_dir <folder>`  : folder with predicted .txt files with bounding boxes  

`-n, --names_file <file>`  : .names file with names of classes   
 
### Results

Following files will be generated:

`metrics.xls` -- table with precision and recall for every class  

`collisions.xls` -- table with collisions of classes done by given detector  

`false_positives.txt` -- list of image file names with false positive detections  

`overlayed_obj_images.txt` -- list of image file names with strongly overlayed objects  

