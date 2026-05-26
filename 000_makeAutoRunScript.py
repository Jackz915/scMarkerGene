import numpy as np
import copy
import os,sys,datetime,time,gzip
import math
import random
import argparse

# print information line
def infoLine(message, infoType="info"):
    
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 1.0",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="infile",    type=str,   required=True, help="Path of input data")
    parser.add_argument("-w", dest="workDIR",   type=str,   required=True, help="Absolute path of work directory")
    parser.add_argument("-t", dest="file_type", choices=["count", "normalized"], default="count", help="Input data type: 'count' or 'normalized'")
    parser.add_argument("--strategy", choices=["upsample", "weights", "none"], default="upsample",  
                        help="Strategy for handling class imbalance. "
                         "Options: upsample (simple oversampling), "
                         "weights (class-weighted loss), "
                         "none (no balancing).")

    args=parser.parse_args()
    
    infile = args.infile # 
    workDIR= args.workDIR #
    file_type = args.file_type # 
    strategy = args.strategy # 
    
    os.system("mkdir -p " + workDIR)
    
    # get absolute path of script
    paths = os.path.split(os.path.abspath(__file__))
    programDIR = paths[0]


    # prepare inputData
    program = programDIR + "/001_prepareData.py"
    cmd =              "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m Prepare input data \033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    
    cmd = cmd + "\n" + "python " + program + " -i " + infile + " -o " + f"{workDIR}/001_prepareData_output" + " -t " + file_type
    cmd = cmd + "\n" + "\n\n\n"

    
    # training network and generating explanations
    program = programDIR + "/002_geneContribution.py"
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m Training network and generating explanations \033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    
    cmd = cmd + "\n" + "python " + program + " -i " + f"{workDIR}/001_prepareData_output" + " -o " + f"{workDIR}/002_geneContribution_output" + f" --strategy {strategy}"
    
    cmd = cmd + "\n" + "\n\n\n"
    
    # make summary on cell marker
    program = programDIR + "/003_extractCellMarker.py"
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m Identifying marker genes in each cell cluster/group \033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    cmd = cmd + "\n" + "python " + program + " -i " + workDIR  + " -e " + infile + " -o " + f"{workDIR}/003_extractCellMarker_output"
    cmd = cmd + "\n" + "\n\n\n"
    
    # end
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m All done! \033[0m\""
    cmd = cmd + "\n" + "echo -e \"\033[32m++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\033[0m\""    
    
    # save cmd to bash
    with open(workDIR + "/autorun.sh", "wt" ) as fo:
        fo.write( cmd + "\n" )
    #
    
    print("Please modify the following bash script as need or directly run the following command")
    print("bash " + workDIR + "/autorun.sh")
#

