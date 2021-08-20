##########################################################################################################
#                                                                                                        #
#    If you run the following code, the program would get a result json from API server of MolGpKa by    #
#    requests package. The result json contains information the atomic numbers of acidic and             # 
#    basic sites in the molecule, predicted pKa values, and the mol-block of the processed molecule,     #
#    which is written by RDKit.                                                                          #
#                                                                                                        #
#    Authors: Xiaolin Pan                                                                                #  
#    2021/01/23                                                                                          #
#                                                                                                        #
##########################################################################################################


import requests
import random
import os
import pandas as pd
from rdkit import Chem

upload_url=r'http://xundrug.cn:5001/modules/upload0/'

def predict_pka(smi):
    param={"Smiles" : ("tmg", smi)}
    headers={'token':'O05DriqqQLlry9kmpCwms2IJLC0MuLQ7'}
    response=requests.post(url=upload_url, files=param, headers=headers)
    jsonbool=int(response.headers['ifjson'])
    if jsonbool==1:
        res_json=response.json()
        if res_json['status'] == 200:
            pka_datas = res_json['gen_datas']
            return pka_datas
        else:
            raise RuntimeError("Error for pKa prediction")
    else:
        raise RuntimeError("Error for pKa prediction")
        
if __name__=="__main__":
    smi = "Clc3ccc2nccc(Cn1cccn1)c2c3"
    data_pka = predict_pka(smi)
    print(data_pka)
