# This script generates a video from a sequence of pictures. Use the Imaging conda env to run.
import pandas as pd
import cv2, os, h5py, sys
sys.path.append(os.path.abspath("RHCVisualisation"))
from RHCImaging.VideoManagment.videolib import imageHiveOverview
from RHCImaging.Preprocessing.preproc import beautify_frame
from RHCImaging.CellContentIdentification.cellcontent import *
from RHCImaging.HiveOpenings.libOpenings import valid_ts
from RHCImaging.libimage import RPiCamV3_img_shape_RGB
from RHCThermalPlots.thermalutil import *
from InfluxDBInterface.libdb import readInfluxCSV
from PIL import Image  # Or OpenCV if preferred
from matplotlib.path import Path
from EasIlastik import * # Just a simple package that runs iLastik in headless mode

# Function to access files and force download
def preload_images(src_path:str):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            # Check for image file extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    # Attempt to open the file
                    with Image.open(file_path) as img:
                        img.verify()  # Ensure file integrity
                    print(f"Preloaded: {file_path}")
                except Exception as e:
                    print(f"Failed to preload {file_path}: {e}")
    
def extractData(csv_path:str, hive:int, timestamps:pd.DatetimeIndex, section_prefix:str="#", verbose:bool=False)->pd.DataFrame:
    '''
    Extracts data from a csv file for a specific hive at specific timestamps.
    '''
    if verbose:
        print(timestamps)
    df = readInfluxCSV(csv_path, section_prefix)
    if verbose:
        print(df)
        print(df.index.dtype)
    relevant_data = df[(df['hive_num'] == hive) & (df.index.isin(timestamps))]
    if relevant_data.empty:
        print(f"No data found for hive {hive} at the timestamps provided!")
        return None
    return relevant_data

def generateMetabolicDF(df:pd.DataFrame)->pd.DataFrame:
    '''
    Generates a df for the metabolic data. Every column is a metabolic measure (ul,ur,ll,lr) and every line is a timestamp.
    parameters:
    - df: pd.DataFrame containing all data for our hive and timestamps. Index should be the timestamps.
    '''
    metabolic_in = df[(df['_measurement'] == 'co2') & (df['_field'] == 'co2')]
    # Remove all negative values
    metabolic_in = metabolic_in[metabolic_in['_value'] >= 0]
    metabolic_out = pd.DataFrame(index=metabolic_in.index.unique())

    column_names = ['ul','ur','ll','lr'] # Upper left, upper right, lower left, lower right
    for column_name in column_names:
        inhive_loc = "upper" if column_name[0] == "u" else "lower"
        if column_name == "ul":
            rpi_num = 1
        elif column_name == "ll":
            rpi_num = 2
        else:
            rpi_num = 3

        metabolic_out[column_name] = metabolic_in[(metabolic_in['inhive_loc'] == inhive_loc) & (metabolic_in['rpi_num'] == rpi_num)]['_value']

    return metabolic_out

def generateHtrDF(df:pd.DataFrame)->list:
    '''
    Generates a list of df for the heater data. Lines of the df are the (same) timestamps, and columns are:
    - status
    - pwm
    - avg_temp
    - obj
    - actuator_instance

    We should thus have 10 rows in the df, and as many items in the list as there are timestamps in df.
    '''
    htr_in = df[(df['_measurement'] == 'htr')]
    timestamps = htr_in.index.unique()
    upper_out = []
    lower_out = []
    for timestamp in timestamps:
        upper_out.append(htr_in[(htr_in.index == timestamp) & (htr_in['inhive_loc'] == 'upper')])
        lower_out.append(htr_in[(htr_in.index == timestamp) & (htr_in['inhive_loc'] == 'lower')])

    # Drop unnecessary columns
    for i in range(len(upper_out)):
        upper_out[i] = upper_out[i].drop(columns=['_measurement'])
        lower_out[i] = lower_out[i].drop(columns=['_measurement'])

    return upper_out, lower_out

def _add_transparent_image(background, foreground, fg_origin:str='top-left', x_offset:int=None, y_offset:int=None):
    '''
    Function adapted from https://stackoverflow.com/a/71701023.
    Background is assumed to have its origin at the top-left corner.
    '''
    if foreground is None:
        return background
    assert fg_origin in ['top-left', 'bottom-left'], f'fg_origin must be "top-left" or "bottom-left". found:{fg_origin}'
    if fg_origin == 'bottom-left':
        new_fg = cv2.flip(foreground, 0)  # Flip vertically
        return _add_transparent_image(background, new_fg, fg_origin='top-left', x_offset=x_offset, y_offset=y_offset)

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background

def putTextRightJustify(
    img,
    text:str,
    bottom_right_coords:tuple,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    color=(0, 0, 0),
    thickness=2,
    line_type=cv2.LINE_AA,
    vertical_align="bottom"  # can be "bottom", "center", or "top"
):
    """
    Draws right-justified text on an OpenCV image.

    Parameters
    ----------
    img : np.ndarray
        Image on which to draw text.
    text : str
        Text to draw.
    bottom_right : tuple(int, int)
        (x, y) coordinates of the bottom-right reference point.
    font : int
        OpenCV font face.
    font_scale : float
        Font scale factor.
    color : tuple(int, int, int)
        Text color in BGR.
    thickness : int
        Thickness of the text stroke.
    line_type : int
        Type of the line (e.g. cv2.LINE_AA).
    vertical_align : str
        "bottom", "center", or "top" vertical alignment relative to the reference point.
    """

    # Measure text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = bottom_right_coords

    # Right-justify: shift left by the text width
    x -= text_width

    # Adjust for vertical alignment
    if vertical_align == "bottom":
        y -= baseline
    elif vertical_align == "center":
        y += text_height // 2 - baseline - 5
    elif vertical_align == "top":
        y += text_height - baseline

    # Draw the text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, line_type)

exp_thermal_shifts = {
    'aSensing1' : {
        1:[(260,510),(260,500),(220,520),(220,420)],
        2:[(260,510),(260,500),(190,440),(210,490)]
    },
    'aSensing2' : {
        1:[(260,520),(260,510),(210,490),(230,370)]
    },
    'aSensing3' : {
        1:[(280,520),(320,430),(230,440),(230,450)]
    },
    'aSensing4' : {
        1:[(240,520),(280,460),(240,440),(180,440)]
    },
    'aSensing3.5' : {
        1:[(280,520),(420,430),(230,440),(180,440)]
    } 
}

class Hive():
    '''
    This class is meant to hold imaging, thermal and metabolic data for a specific hive at a specific time. It contains the following data:
    - 4 images of the hive (hxr1, hxr2, hxr3, hxr4)
    - 2 ThermalFrames objects (upper, lower)
    - 4 metabolic measures (right, left, upper right, upper left)
    '''

    # Some class variables
    resize_factor = 10.1 # Resize factor for the thermal images relative to the IR images
    inter_htr_dist = 25 # Distance between heaters in pixels
    htr_size=(800,800) # Size of the heaters in pixels (width, height)
    base_thermal_shifts = [[(260,510),(260,500),(220,520),(220,420)], # Hive 1
                           [(260,510),(260,500),(190,440),(220,490)]] # Hive 2
    base_co2_pos = {'ul':(300,380),'ur':(4350,380),'ll':(330,380),'lr':(4350,380)}
    
    @staticmethod
    def process_ilastik_mask(honey_mask, hive_num:int, rpi_num:int, min_size:int=3000,threshold:int =128):
        #TODO: I think the lower is only true for aSensing1 hive1… Remove ?
        if hive_num == 1:
            if rpi_num == 2:
                honey_mask[-30:,:] = 0            
            elif rpi_num == 4:
                honey_mask[-70:,:] = 0

        mask = thresholding(honey_mask, threshold)

        mask=morph(mask, kernel_size=7, close_first=False)
        # Remove the pixel patches that are smaller than a certain size
        mask=remove_small_patches(mask, min_size)
        return mask

    def __init__(self, ts:pd.Timestamp, imgs:list, imgs_preprocessed:bool, imgs_names:list[str], upper:ThermalFrame = None, lower:ThermalFrame = None, metabolic:pd.DataFrame = None, htr_upper:pd.DataFrame = None, htr_lower:pd.DataFrame = None, hive_nb:int = 0):
        '''
        Constructor for the Hive class.
        Parameters:
        - ts: pd.Timestamp, datetime of the data
        - imgs: list of 4 images of the hive (hxr1, hxr2, hxr3, hxr4)
        - imgs_preprocessed: bool, whether the images have been preprocessed already or not
        - imgs_names: list of 4 strings, names of the images
        - upper: ThermalFrame object for the upper part of the hive
        - lower: ThermalFrame object for the lower part of the hive
        - metabolic: pd.DataFrame that has ['ul','ur','ll','lr'] as columns and the metabolic measures as values
        - htr_upper: pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        - htr_lower: pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        - hive_nb: int, hive number (1 or 2). 0 if unknown. Used to verify data validity, set the name of the hive and set the thermal shifts.
        '''

        if len(imgs) != 4 or len(imgs_names) != 4:
            raise ValueError("imgs must contain 4 images")
        if metabolic is not None and len(metabolic) != 4:
            raise ValueError("metabolic must contain 4 values")
        
        self.ts = ts
        self.hive_nb = hive_nb
        if self.hive_nb == 1 or self.hive_nb == 2 or self.hive_nb == 3: # Hive 4 and 5 can only be Graz hives…
            self.valid = valid_ts(ts, self.hive_nb, recovery_time=180) # We consider 180' for RHCVisu hives
        else:
            self.valid = True # We assume the data is valid if hive number is not known

        self.imgs = imgs
        if imgs_preprocessed:
            self.pp_imgs = imgs
        else:
            self.pp_imgs = None
        self.imgs_names = imgs_names
        self.name = f"h{self.hive_nb}_{ts.strftime('%y%m%d-%H%M%Z')}"

        if self.hive_nb == 1 or self.hive_nb == 2: # No thermal shifts for h3 as box hive
            self.setThermalShifts(Hive.base_thermal_shifts[self.hive_nb-1]) # Set the thermal shifts for the hive number
        else:
            self.setThermalShifts(Hive.base_thermal_shifts[0])
        self.upper_tf = upper
        if self.upper_tf is not None:
            self.upper_tf.calculate_thermal_field()
        self.lower_tf = lower
        if self.lower_tf is not None:
            self.lower_tf.calculate_thermal_field()
        self.metabolic = metabolic # pd.DataFrame that has ['ul','ur','ll','lr'] as columns and the metabolic measures as values
        self.htr_upper = htr_upper # pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        self.htr_lower = htr_lower # pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        # To store the pixel shifts between the thermal and imaging data. A list of 4 tuples, each tuple containing the x,y shifts for the corresponding RPi image.
        self.co2_pos = Hive.base_co2_pos

    def computeHtrPos(self):
        '''
        Computes the positions of the heaters in pp_imgs based on self.thermal_shifts.
        affects: 
        - self.htr_pos: dict containing the positions of the heaters for each rpi (top-left and bottom-right corners of the rectangle). The keys are the rpi numbers (0,1,2,3), followed by the heater number (h00 to h09).
        NOTE: positions are for images that have already been flipped horizontally.
        '''
        htr_pos = {}
        for i in range(4):
            htr_pos[i] = {}
            for j in range(10):
                if i < 2:
                    x_pos = self.thermal_shifts[i][0] + self.inter_htr_dist + (4-j//2) * (self.inter_htr_dist + self.htr_size[0])
                else:
                    x_pos = self.thermal_shifts[i][0] + self.inter_htr_dist + (j//2) * (self.inter_htr_dist + self.htr_size[0])

                y_pos = self.thermal_shifts[i][1] + 2*self.inter_htr_dist + (j%2) * (3*self.inter_htr_dist + self.htr_size[1])
                htr_pos[i][f'h{j:02d}'] = ((x_pos, y_pos),(x_pos+self.htr_size[0],y_pos+self.htr_size[1]))
        self.htr_pos = htr_pos

    def computeHtrHoneyContent(self, verbose:bool=False):
        '''
        This function computes the honey content of all heaters of all rpis based on the honey masks.
        returns:
        - htr_content: dict containing the honey content for each heater for each rpi. The keys are the rpi numbers (0,1,2,3), followed by the heater number (h00 to h09).
        affects:
        - sets self.htr_content, in percentages, for each heater in each rpi.
        - sets self.frame_content, in percentages, for each heater in the upper and lower frame.
        - sets self.frame_content_ml, in ml, for each heater in the upper and lower frame.
        '''
        if self.honey_masks is None:
            raise ValueError("Honey masks not loaded. Use loadHoneyMasks() to load the masks or generate them.")
        
        if self.htr_pos is None:
            self.computeHtrPos()

        htr_content = {}
        for rpi in range(4):
            htr_content[rpi+1] = {}
            mask = self.honey_masks[rpi]
            if mask is None:
                continue
            for htr in range(10):
                htr_pos = self.htr_pos[rpi][f'h{htr:02d}']
                htr_pos = ((htr_pos[0][0] - self.thermal_shifts[rpi][0], htr_pos[0][1] - self.thermal_shifts[rpi][1]), (htr_pos[1][0] - self.thermal_shifts[rpi][0], htr_pos[1][1] - self.thermal_shifts[rpi][1]))

                # Get the honey mask in the area of the heater
                mask_honey = mask[htr_pos[0][1]:htr_pos[1][1], htr_pos[0][0]:htr_pos[1][0]]
                # Compute the honey content
                htr_content[rpi+1][f'h{htr:02d}'] = (np.sum(mask_honey>0) / (mask_honey.shape[0] * mask_honey.shape[1])) * 100 # In percents
        
        self.htr_content = htr_content
        frame_content = {}
        frame_content_ml = {}
        for ihl in ["upper", "lower"]:
            frame_content[ihl] = {}
            frame_content_ml[ihl] = {}
            if ihl == "upper" and (self.htr_content[1] == {} or self.htr_content[3] == {}):
                continue
            if ihl == "lower" and (self.htr_content[2] == {} or self.htr_content[4] == {}):
                continue
            for htr in range(10):
                frame_content[ihl][f'h{htr:02d}'] = np.mean([self.htr_content[1][f'h{htr:02d}'], self.htr_content[3][f'h{htr:02d}']]) if ihl == "upper" else np.mean([self.htr_content[2][f'h{htr:02d}'], self.htr_content[4][f'h{htr:02d}']])
                frame_content_ml[ihl][f'h{htr:02d}'] = frame_content[ihl][f'h{htr:02d}'] * 0.9 # Convert to ml (assuming 100% is 90ml)
        self.frame_content = frame_content
        self.frame_content_ml = frame_content_ml

        if verbose:
            print("Heater content:")
            print(self.htr_content)
            print("Frame content:")
            print(self.frame_content)
            print("Frame content in ml:")
            print(self.frame_content_ml)
        
        return self.htr_content

    def setCo2Pos(self, co2_pos:dict):
        '''
        Sets the position of the CO2 measures on the images.
        '''
        self.co2_pos = co2_pos

    def setThermalShifts(self, thermal_shifts:list):
        '''
        Sets the pixel shifts between the thermal and imaging data.
        '''
        assert len(thermal_shifts) == 4, "thermal_shifts must contain 4 tuples"
        for shift in thermal_shifts:
            assert len(shift) == 2, "Each tuple in thermal_shifts must contain 2 values"

        self.thermal_shifts = thermal_shifts
        # Re-compute the heater positions
        self.computeHtrPos()

    def getBeeArena(self):
        '''
        Returns a list that contains the bee arena (rectangle coordinates: starting in thermal_shifts and of size ThermalFrame.x_pcb*self.resize_factor) for each image
        '''
        bee_arenas = []
        for i in range(4):
            bee_arena = ((self.thermal_shifts[i][0],self.thermal_shifts[i][1]),(int(self.thermal_shifts[i][0]+ThermalFrame.x_pcb*self.resize_factor),int(self.thermal_shifts[i][1]+ThermalFrame.y_pcb*self.resize_factor)))
            bee_arenas.append(bee_arena)

        return bee_arenas
    
    def getBeeArenaImages(self):
        '''
        Returns the images of the bee arenas for each RPi. The images are cropped to the size of the bee arenas.
        '''
        # Check if self.pp_imgs is None are computed or not. Compute if not
        if self.pp_imgs is None:
            self.pp_imgs = []
            for rpi in range(4):
                if self.imgs[rpi] is None:
                    self.pp_imgs.append(None)
                else:
                    self.pp_imgs.append(beautify_frame(self.imgs[rpi]))

        bee_arena_px = self.getBeeArena()
        bee_arenas_imgs = []
        for rpi in range(4):
            if self.pp_imgs[rpi] is None:
                bee_arenas_imgs.append(None)
            else:
                bee_arenas_imgs.append(self.pp_imgs[rpi][bee_arena_px[rpi][0][1]:bee_arena_px[rpi][1][1], bee_arena_px[rpi][0][0]:bee_arena_px[rpi][1][0]])
        
        return bee_arenas_imgs
    
    def ilastikSegmentHoney(self, model_path:str, rpis:list[int]=[1,2,3,4], verbose:bool=False):
        '''
        Uses the provided ilastik model to segment the honey in the images. Uses a tmp folder to store the cropped images but deletes them after processing.
        args:
        - model_path: str, path to the ilastik model
        - rpis: list of int, rpi cameras to process. Default is all ([1,2,3,4])
        '''
        cropped_images = self.getBeeArenaImages()
        # Create a temporary directory to store the cropped images
        dir_name = f"tmp_{self.name}"
        tmp_dir = os.path.join(os.getcwd(), dir_name)
        os.makedirs(tmp_dir, exist_ok=True)
                
        cropped_images_dir = os.path.join(tmp_dir, "cropped_images/")
        results_folder = os.path.join(tmp_dir, "results/")
        os.makedirs(cropped_images_dir, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        # Save the cropped images to the temporary directory
        for i, img in enumerate(cropped_images):
            if i+1 in rpis:
                img_name = self.imgs_names[i]
                img_path = os.path.join(cropped_images_dir, f"{img_name}_cropped.png")
                cv2.imwrite(img_path, img)

        # Run ilastik on the temporary directory
        assert model_path.endswith(".ilp"), "Model path must end with .ilp"
        if verbose:
            print("Results folder: ", results_folder)
            print(tmp_dir,model_path,results_folder)
            
        run_ilastik(input_path = cropped_images_dir,
                    model_path = model_path,
                    result_base_path = results_folder,
                    export_source = "Probabilities",
                    output_format = "hdf5",
                    verbose = verbose)
        
        result_files = [f for f in os.listdir(results_folder) if f.endswith(".h5")]
        assert len(result_files) == len(rpis), "The number of result files does not match the number of requested processed images."
        result_files.sort()
        masks = []

        for i in range(4):
            if i+1 not in rpis:
                masks.append(None)
                continue
            # Get the file name corresponding to the current rpi. It should contain "rpi{i+1}" in the name.
            file = [f for f in result_files if f"rpi{i+1}" in f][0]
            hive_num = int(file.split("_")[0][4:]) # Get the hive number from the file name
            mask_path = os.path.join(results_folder, file)
            # Open file in read mode
            with h5py.File(mask_path, "r") as f:
                honey_mask = f["exported_data"][:, :, 0]
                mask = Hive.process_ilastik_mask(honey_mask,hive_num=hive_num, rpi_num=i+1, min_size=2500, threshold=40)
                # Append the result to the list
                masks.append(mask)

        self.loadHoneyMasks(masks)

        # Delete tmp_dir and its contents
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        os.rmdir(tmp_dir)

    def _processRegionGrownMask(self, mask):
        # Convert the mask to a binary mask
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask = morph(mask, kernel_size=3,close_first=False)
        mask = remove_small_patches(mask, min_size=10000) # Remove small patches that are not honey
        mask = morph(mask, kernel_size=5, close_first=True)
        mask = remove_small_patches(mask, min_size=10000) # Remove small patches that are not honey
        return mask

    def regionGrowSegmentHoney(self, rpis:list[int]=[1,2,3,4], gradient_threshold:int=4, value_threshold:int=160, min_size:int=700, verbose:bool=False):
        '''
        Uses the region growing algorithm to segment the honey in the images. Uses a tmp folder to store the cropped images but deletes them after processing.
        args:
        - gradient_threshold: int, threshold for the gradient to start the region growing algorithm
        - value_threshold: int, threshold for the pixel value to start the region growing algorithm
        - min_size: int, minimum size of the region to be considered as honey
        - rpis: list of int, rpi cameras to process. The rpis not included will be given a 0-mask (no honey anywhere). Default is all ([1,2,3,4]).
        
        Sets (and overwrites) self.honey_masks.
        '''
        cropped_images = self.getBeeArenaImages()
        masks = []
        for i, img in enumerate(cropped_images):
            if i+1 not in rpis:
                masks.append(None)
                continue
            # Apply the region growing algorithm to the image
            mask = region_growing(img, gradient_threshold=gradient_threshold, value_threshold=value_threshold, min_size=min_size, verbose=verbose)
            masks.append(mask)

        empty_mask = np.zeros_like(cropped_images[0], dtype=np.uint8)  # Create an empty mask for rpis not processed
        for i, mask in enumerate(masks):
            if mask is None:
                masks[i] = empty_mask
            else:
                masks[i] = self._processRegionGrownMask(mask)

        self.loadHoneyMasks(masks)

    def _co2_snapshot(self,rgb_imgs:list):
        min_size = 4
        max_size = 10

        for i, img in enumerate(rgb_imgs):
            if i == 0 or i == 2:
                co2_showed = ['ur','ul']
            else:
                co2_showed = ['lr','ll']

            co2_pos = self.co2_pos
            if i >= 2:
                # Flip the co2_pos horizontally
                co2_pos = {k:(img.shape[1]-v[0]-220, v[1]) for k,v in co2_pos.items()}

            for co2 in co2_showed:
                # Compute the size of the text based on the metabolic measure. Put max_size at 30000 and min_size at 300, linearly.
                # First clip the values to be between 360 and 30000
                if self.metabolic[co2] is None or np.isnan(self.metabolic[co2]):
                    continue
                co2_value = int(np.clip(self.metabolic[co2],360,30000))
                color = (255 * (co2_value - 360) / (30000 - 360),0,0)
                size = min_size + (max_size - min_size) * (co2_value - 360) / (30000 - 360)
                if (co2[1] == 'r' and i<2) or (co2[1] == 'l' and i>=2):
                    # Change the x value of co2_pos to decrease with the size
                    co2_pos[co2] = (co2_pos[co2][0] - int(800*(size-min_size)/(max_size-min_size)), co2_pos[co2][1])
                cv2.putText(rgb_imgs[i], f"{co2_value}", co2_pos[co2], cv2.FONT_HERSHEY_SIMPLEX, size, color, 20, cv2.LINE_AA)

    def _tmp_snapshot(self, rgb_imgs:list, v_min, v_max, thermal_transparency, contours:list, annotate_contours:bool):
        overlays = [None, None]
        # Store the max temp coordinates and value
        max_temp = -100
        min_temp = 200
        max_temp_coords = (0,0,0) # First coordinate is the frame, then x and y
        for i, tf in enumerate([self.upper_tf, self.lower_tf]):
            if tf is not None:
                if tf.min_temp < min_temp:
                    min_temp = tf.min_temp
                if tf.max_temp > max_temp:
                    max_temp = tf.max_temp
                    max_temp_coords = (i,)+ tf.get_max_temp_pos(origin='upper')
                therm_field_norm = (tf.thermal_field - v_min) / (v_max - v_min)
                # Apply matplotlib colormap (e.g., 'bwr')
                colormap = plt.colormaps['bwr']
                overlay_colored = colormap(therm_field_norm)  # Returns RGBA values in [0, 1]
                # Scale to [0, 255] for OpenCV compatibility
                overlay_rgb = (overlay_colored * 255).astype(np.uint8)
                overlay_rgb[:,:,3] = int(255*thermal_transparency)
                overlay_rgb = cv2.resize(overlay_rgb, (int(overlay_rgb.shape[1] * Hive.resize_factor), int(overlay_rgb.shape[0] * Hive.resize_factor)), interpolation=cv2.INTER_NEAREST)
                overlays[i] = overlay_rgb

        fig, ax = plt.subplots()  # Create a dummy figure to prevent automatic plotting
        _contours = [None, None]
        for i,tf in enumerate([self.upper_tf, self.lower_tf]):
            if tf is not None:
                _contours[i] = ax.contour(tf.thermal_field, levels=contours, colors='none') # Only compute, no color
        if annotate_contours:
            labels = [None, None]
            for i, cs in enumerate(_contours):
                if cs is not None:
                    labels[i] = ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f"{x:.0f} C")
        else:
            labels = []

        plt.close(fig)  # Close the figure to prevent display

        # Extract paths from the contour
        paths = [cs.get_paths() if cs is not None else None for cs in _contours]

        # Create a blank canvas for OpenCV drawing
        canvas = [np.zeros_like(overlay[:,:,0]) if overlay is not None else None for overlay in overlays]

        tf_shape = self.upper_tf.thermal_field.shape if self.upper_tf is not None else self.lower_tf.thermal_field.shape
        # Draw the contours onto the canvas
        for i, frame_paths in enumerate(paths): # For each frame
            if frame_paths is None:
                continue
            for path in frame_paths: # For each level in the frame
                if path.vertices.size == 0:
                    continue
                # Get the vertices and codes
                vertices = path.vertices
                codes = path.codes

                # Initialize a list to store points for each disjoint segment
                segment_points = []

                for vert, code in zip(vertices, codes):
                    if code == Path.MOVETO:
                        # Start of a new segment: draw the previous segment if it exists
                        if segment_points:
                            # Scale and convert to integer pixel coordinates
                            segment_points = np.array(segment_points)
                            segment_points[:, 0] *= canvas[i].shape[1] / tf_shape[1]
                            segment_points[:, 1] *= canvas[i].shape[0] / tf_shape[0]
                            segment_points = np.round(segment_points).astype(np.int32)
                            
                            # Draw the segment as a polyline
                            cv2.polylines(canvas[i], [segment_points], isClosed=False, color=255, thickness=5)
                        
                        # Start a new segment
                        segment_points = [vert]
                    elif code in (Path.LINETO, Path.CLOSEPOLY):
                        # Continue the current segment
                        segment_points.append(vert)
                
                # Draw the last segment if it exists
                if segment_points:
                    segment_points = np.array(segment_points)
                    segment_points[:, 0] *= canvas[i].shape[1] / tf_shape[1]
                    segment_points[:, 1] *= canvas[i].shape[0] / tf_shape[0]
                    segment_points = np.round(segment_points).astype(np.int32)
                    cv2.polylines(canvas[i], [segment_points], isClosed=False, color=255, thickness=5)

        # Overlay the contours onto your RGB overlay
        for i, _ in enumerate(overlays):
            if canvas[i] is not None:
                overlays[i][canvas[i] > 0] = (0, 0, 0, 255)  # Black contour with full opacity

        # Put a circular marker at the max temperature coordinates
        cv2.circle(
            overlays[max_temp_coords[0]], 
            (int(max_temp_coords[1] * Hive.resize_factor), int(max_temp_coords[2] * Hive.resize_factor)), 
            20, 
            (255, 0, 0, 255), 
            -1
        )

        # Prepare overlays for each picture
        overlays_flipped = {0:overlays[0],
                            1:overlays[1],
                            2:cv2.flip(overlays[0],1) if overlays[0] is not None else None,
                            3:cv2.flip(overlays[1],1) if overlays[1] is not None else None}
        
        for i, bg in enumerate(rgb_imgs):
            if overlays_flipped[i] is None:
                continue
            overlay_rgb = overlays_flipped[i]
            # Add the max temperature value at the max temperature coordinates:
            if i%2 == max_temp_coords[0]:
                coords = (int(max_temp_coords[1] * Hive.resize_factor), int(max_temp_coords[2] * Hive.resize_factor))
                if i>=2:
                    # Flip the coordinates horizontally
                    coords = (overlay_rgb.shape[1]-coords[0],coords[1])
                # Add a 20px margin to the coordinates
                coords = (coords[0]+20,coords[1]-20)
                cv2.putText(overlay_rgb, f"{max_temp:.1f}", coords, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0, 255), 10, cv2.LINE_AA)
            
            if labels != []:
                # Add the corresponding label to the contours
                img_labels = labels[i if i<2 else i-2]
                for txt in img_labels:
                    if txt.get_text() == '':
                        continue
                    x, y = txt.get_position()
                    x = int(x * overlay_rgb.shape[1] / tf_shape[1])
                    y = int(y * overlay_rgb.shape[0] / tf_shape[0])
                    if i >= 2:
                        x = overlay_rgb.shape[1] - x
                    cv2.putText(overlay_rgb, txt.get_text(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3, cv2.LINE_AA)
            rgb_imgs[i]= _add_transparent_image(bg, overlay_rgb, x_offset=self.thermal_shifts[i][0], y_offset=self.thermal_shifts[i][1])
        
        return rgb_imgs, min_temp
    
    def _htr_snapshot(self,rgb_bg:list, show_obj:bool=True):
        # Draw a rectangle around the heaters and add information about the heaters
        for i, _ in enumerate(rgb_bg):
            htrs = self.htr_upper if (i == 0 or i == 2) else self.htr_lower
            for htr in [f'h{i:02d}' for i in range(10)]:
                htr_df = htrs[htrs['actuator_instance']==htr]
                pwm = htr_df[htr_df['_field']=='pwm']['_value'].values[0]
                obj = htr_df[htr_df['_field']=='obj']['_value'].values[0]

                if pwm > 0 or obj > 0:
                    # Draw a rectangle around the heater
                    color = (255 * pwm / 950,0,0)
                    width = int(4 + 7 * pwm / 950)
                    mrg = 10 # Just a small padding around the text

                    cv2.rectangle(rgb_bg[i], self.htr_pos[i][htr][0], self.htr_pos[i][htr][1], color, width)
                    # Put pwm bottom left
                    cv2.putText(rgb_bg[i], f"{int(pwm)}", (self.htr_pos[i][htr][0][0]+mrg,self.htr_pos[i][htr][1][1]-mrg), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5, cv2.LINE_AA)
                    if show_obj:
                        # Put obj bottom right
                        putTextRightJustify(rgb_bg[i], f"{int(obj)} C", (self.htr_pos[i][htr][1][0]-mrg,self.htr_pos[i][htr][1][1]-mrg), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5, cv2.LINE_AA, vertical_align="center")
                    # Put the heater number on top left of the rectangle
                    cv2.putText(rgb_bg[i], htr, (self.htr_pos[i][htr][0][0]+mrg,self.htr_pos[i][htr][0][1]+10*mrg), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5, cv2.LINE_AA)

    def snapshot(self, thermal_transparency:float=0.25, v_min:float=10, v_max:float=35, contours:list=[], annotate_contours:bool=False, annotate_names:bool=True, show_frame_border:bool=False, show_htr_obj:bool=True, check_validity:bool=True, use_cet_time:bool=False):
        '''
        Generates a global image with the 4 images of the hives with the timestamp on the pictures. It then adds the ThermalFrames ontop of the images.
        '''
        # Preprocess images if not already done
        if self.pp_imgs is None:
            # Preprocess images with Preprocessing library
            self.pp_imgs = []
            for img in self.imgs:
                if img is None:
                    self.pp_imgs.append(None)
                else:
                    self.pp_imgs.append(beautify_frame(img))

        black_image_rgb = np.zeros(RPiCamV3_img_shape_RGB, dtype=np.uint8)
        rgb_bg = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img is not None else black_image_rgb.copy() for img in self.pp_imgs]
        
        min_temp = -273
        if self.upper_tf is not None or self.lower_tf is not None:
            rgb_bg, min_temp = self._tmp_snapshot(rgb_bg,v_min,v_max,thermal_transparency,contours, annotate_contours=annotate_contours) # Adds thermal field and isotherms to the images

        if self.htr_upper is not None and self.htr_lower is not None:
            self._htr_snapshot(rgb_bg, show_obj=show_htr_obj) # Adds heaters data on the images

        if self.metabolic is not None:
            self._co2_snapshot(rgb_bg) # Add the CO2 measurements on the images

        if show_frame_border:
            # Draw a rectangle around the hive using self.thermal_shifts
            for i, img in enumerate(rgb_bg):
                # Draw a rectangle around the hive
                rectangles = self.getBeeArena()
                cv2.rectangle(img, rectangles[i][0], rectangles[i][1], (255, 0, 0), 10)

        if annotate_names:
            assembled_img = imageHiveOverview(rgb_bg, rgb=True, img_names=self.imgs_names, dt=self.ts, use_cet_time=use_cet_time, valid=(self.valid or not check_validity))
        else:
            assembled_img = imageHiveOverview(rgb_bg, rgb=True, dt=self.ts, use_cet_time=use_cet_time, valid=(self.valid or not check_validity))

        # add ambient temperature on the image (min temp)
        ambient_t_text = f"Ambient: {min_temp:.1f} C"
        (text_width, text_height), _ = cv2.getTextSize(ambient_t_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        rectangle_bgr = (255, 255, 255)
        box_coords = ((2700, 2130 + 15), (2700 + text_width, 2130 - text_height - 15))
        # Add ambient temperature to the image
        cv2.rectangle(assembled_img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(assembled_img, ambient_t_text, (2700, 2130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

        return assembled_img
    
    def honeySnapshot(self, transparency:float=0.25, v_min:float=10, v_max:float=35, contours:list=[], annotate_names:bool=True, show_frame_border:bool=False, show_htrs:bool=False):
        '''
        Generates a global image with the 4 images of the hives with the timestamp on the pictures. It then adds the honey segmentation masks ontop of the images.
        '''
        assert self.honey_masks is not None, "Honey masks not loaded. Use loadHoneyMasks() to load the masks or generate them."
        assert len(self.honey_masks) == 4, "Honey masks must contain 4 images"
        assert self.pp_imgs is not None, "Images not preprocessed. Use snapshot() to preprocess the images first."

        black_image_rgb = np.zeros(RPiCamV3_img_shape_RGB, dtype=np.uint8)
        rgb_bg = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img is not None else black_image_rgb.copy() for img in self.pp_imgs]
        if show_frame_border:
            # Draw a rectangle around the hive using self.thermal_shifts
            for i, img in enumerate(rgb_bg):
                # Draw a rectangle around the bee arena
                rectangles = self.getBeeArena()
                cv2.rectangle(img, rectangles[i][0], rectangles[i][1], (255, 0, 0), 10)

        # Convert the masks to RGB in yellow
        masks_rgba = []
        for mask in self.honey_masks:
            if mask is None:
                masks_rgba.append(None)
                continue
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[mask == 255] = (255, 255, 0, int(255*transparency))
            masks_rgba.append(mask_rgba)

        # Overlay the honey masks on the images
        for i, img in enumerate(rgb_bg):
            if masks_rgba[i] is not None:
                rgb_bg[i] = _add_transparent_image(img, masks_rgba[i], x_offset=self.thermal_shifts[i][0], y_offset=self.thermal_shifts[i][1])
        
        if show_htrs and self.htr_pos is not None:
            self._htr_snapshot(rgb_bg) # Adds heaters data on the images

        if annotate_names:
            assembled_img = imageHiveOverview(rgb_bg, rgb=True, img_names=self.imgs_names, dt=self.ts, valid=self.valid)
        else:
            assembled_img = imageHiveOverview(rgb_bg, rgb=True, dt=self.ts, valid=self.valid)
        return assembled_img

    def loadHoneyMasks(self, masks:list):
        '''
        Loads the honey masks for each image. The masks are stored in a list of 4 images (1 for each RPi).
        The masks should be in the same order as the images.
        '''
        assert len(masks) == 4, "Masks must contain 4 images"
        # make sure at least one mask is not None
        assert any(mask is not None for mask in masks), "At least one mask must be provided"
        # If mask is binary, convert it to uint8
        for i, mask in enumerate(masks):
            if mask is not None and np.max(mask) <= 1:
                    masks[i] = (mask * 255).astype(np.uint8)
                    
        x,y = ThermalFrame.x_pcb*self.resize_factor, ThermalFrame.y_pcb*self.resize_factor
        for i, mask in enumerate(masks):
            if mask is not None:     
                assert mask.shape[0] == y, f"Mask {i} has a height of {mask.shape[0]} instead of {y}"
                assert mask.shape[1] == x, f"Mask {i} has a width of {mask.shape[1]} instead of {x}"
                assert mask.dtype == np.uint8, f"Mask {i} has a dtype of {mask.dtype} instead of np.uint8"
            
        self.honey_masks = masks

    def correctHoneyMasks(self, corrections:list, neg_mask:bool):
        '''
        Corrects the honey masks by applying a logical AND (negative mask) or OR (positive mask) operation between the masks and the corrections.

        :param corrections: list of 4 images, corrections to apply to the honey masks. 
                        The images should be binary and have the same size as the honey masks. 
                        Use None for RPi images that should not be corrected.
        :param neg_mask: bool, if True, the correction masks are considered as negative masks (i.e., areas to remove from the honey masks). Else,
                        they are considered as positive masks (i.e., areas to force to the honey masks).
        '''
        assert len(corrections) == len(self.honey_masks), "You must provide a correction for each RPi image. Use None for RPi images that do not need correction."
        # Ensure each correction has the same size as the corresponding honey mask
        for i, correction in enumerate(corrections):
            if correction is not None:
                assert correction.shape == self.honey_masks[i].shape, f"Correction for RPi {i+1} has a different size than the honey mask: {correction.shape} vs {self.honey_masks[i].shape}"
                assert correction.dtype == np.uint8, f"Correction for RPi {i+1} has a dtype of {correction.dtype} instead of np.uint8"
                # Ensure the correction is binary
                if np.max(correction) <= 1:
                    corrections[i] = (correction * 255).astype(np.uint8)
        for i, mask in enumerate(self.honey_masks):
            if mask is not None and corrections[i] is not None:
                if neg_mask:
                    self.honey_masks[i] = cv2.bitwise_and(mask, corrections[i])
                else:
                    self.honey_masks[i] = cv2.bitwise_or(mask, corrections[i])
            elif corrections[i] is not None:
                raise ValueError(f"Correction for RPi {i+1} is not None but the mask is None. Please check your inputs.")

    def compute_pixel_shifts(self):
        '''
        Computes the pixel shifts between the thermal and the imaging data, for every image of the hive.
        Returns a list of 4 tuples, each tuple containing the x,y shifts for the corresponding RPi image.
        NOTE: This function is currently not used as the line detection method is not reliable.
        '''
        shifts = []
        if self.pp_imgs is None:
            # Preprocess images with Preprocessing library
            self.pp_imgs = []
            for img in self.imgs:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.pp_imgs.append(beautify_frame(gray_img))

        for pp_img in self.pp_imgs:
            # Perform edge detection
            edges = cv2.Canny(pp_img, 350, 450)
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=3500, maxLineGap=1000)
            h_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 20] # Horizontal lines
            v_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 20] # Vertical lines
            # Discard the v_lines that are too close to the left border
            if v_lines is not None:
                v_lines = [line for line in v_lines if line[0][0] > 10]

            if len(h_lines) == 0:
                raise ValueError("No horizontal lines detected")
            if len(v_lines) == 0:
                raise ValueError("No vertical lines detected")
            
            lowest_h_line = h_lines[np.argmax([line[0][1] for line in h_lines])]
            leftest_v_line = v_lines[np.argmin([line[0][0] for line in v_lines])]

            upper_shift = np.mean([lowest_h_line[0][1], lowest_h_line[0][3]])
            left_shift = np.mean([leftest_v_line[0][0], leftest_v_line[0][2]])
            shifts.append((left_shift, upper_shift))
        
        self.thermal_shifts = shifts
        return self.thermal_shifts
