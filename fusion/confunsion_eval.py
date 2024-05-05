
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from datetime import datetime
def confunsionevl():


    rootpath= os.path.dirname(os.path.abspath(__file__))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file = open(os.path.join(os.path.dirname(rootpath), "evalresults.txt"), "a")
    file.write(current_time + '\n')
    # for _data_name in ['COD10K']:
    for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:

        mask_root = os.path.join(os.path.dirname(rootpath),'Dataset/TestDataset/{}/GT'.format(_data_name))
        pred_root = os.path.join(rootpath,'fusionresult/{}/'.format( _data_name))
        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "Smeasure": round(sm,4) ,
            "meanEm": round(em["curve"].mean(), 4),
            "wFmeasure": round(wfm,4),
            "MAE": round(mae,4),
            "maxEm": round(em["curve"].max(),4),
            "meanFm": round(fm["curve"].mean(),4),
            # "maxFm": round(fm["curve"].max(),4),
            # "adpFm": round(fm["adp"],4),
            # "adpEm": round(em["adp"],4),
        }

        print(results)

        file.write(_data_name+' '+str(results)+'\n')
    file.close()
if __name__=="__main__":
    confunsionevl()