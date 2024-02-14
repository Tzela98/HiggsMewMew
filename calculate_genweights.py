import json
import subprocess
import time
import uproot
from rich.progress import Progress
import numpy as np


def read_filelist_from_das(dbs):
    filedict = {}
    das_query = "file dataset={}".format(dbs)
    das_query += " instance=prod/global"
    cmd = [
        "/cvmfs/cms.cern.ch/common/dasgoclient --query '{}' --json".format(das_query)
    ]
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    jsonS = output.communicate()[0]
    filelist = json.loads(jsonS)
    for file in filelist:
        filedict[file["file"][0]["name"]] = file["file"][0]["nevents"]
    return [
        "{prefix}/{path}".format(prefix="root://cms-xrd-global.cern.ch/", path=file)
        for file in filedict.keys()
    ]


def calculate_genweight_uproot(dataset):
    print(f"Counting negative and positive genweights for {dataset['nick']}...")
    filelist = read_filelist_from_das(dataset["dbs"])
    negative = 0
    positive = 0
    # set a threshold that if more than 10% of the files fail, the function returns None
    threshold = len(filelist) // 10
    fails = 0

    print(f"Threshold for failed files: {threshold}")
    print(f"Number of files: {len(filelist)}")
    # loop over all files and count the number of negative and positive genweights
    with Progress() as progress:
        task = progress.add_task("Files read ", total=len(filelist))
        filelist = [file + ":Events" for file in filelist]
        for i, file in enumerate(filelist):
            try:
                events = uproot.open(file, timeout=5)
                array = events["genWeight"].array(library="np")
                negative += np.count_nonzero(array < 0)
                positive += np.count_nonzero(array >= 0)
                # print(f"File {i+1}/{len(filelist)} of {dataset['nick']} read")
                progress.update(task, advance=1)
            except Exception as e:
                print("Error when reading input file")
                print(e)
                fails += 1
            if fails > threshold:
                print("Too many files failed, returning None")
                return None
        print(f"Negative: {negative} // Positive: {positive}")
        negfrac = negative / (negative + positive)
        genweight = 1 - 2 * negfrac
        print(f"Final genweight: {genweight}")
        return genweight


def main():
    dataset_info = {
        "nick": "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext2",
        "dbs": "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1/NANOAODSIM"
    }
    calculate_genweight_uproot(dataset_info)


if __name__ == "__main__":
    main()

