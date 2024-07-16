import pandas as pd

def filter_muons_from_csv(file_path):
    df = pd.read_csv(file_path)
    filtered_df = df[df['nmuons'] <= 3]
    return filtered_df


def main():
    csv_paths = [
        '/work/ehettwer/HiggsMewMew/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
        '/work/ehettwer/HiggsMewMew/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106XZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
        '/work/ehettwer/HiggsMewMew/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
        '/work/ehettwer/HiggsMewMew/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ]

    for path in csv_paths:
        filtered_df = filter_muons_from_csv(path)
        print(filtered_df.head())
        print(f"Number of rows before filtering: {len(pd.read_csv(path))}")
        print(f"Number of rows after filtering: {len(filtered_df)}")

        filtered_df.to_csv(path.replace('.csv', '_filtered.csv'), index=False)

if __name__ == "__main__":
    main()