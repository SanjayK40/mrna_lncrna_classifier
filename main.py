from Components.preprocess import process
from Components.utils import Utility
from Components.trainer import Trainer
from Components.plot import Plotter
from Components.evaluate import evaluate
import os
from yaml_loader import load_yaml
from Components.data_loader import loader
config=load_yaml("config.yaml")

if __name__=='__main__':
    processor=process(config)
    load=loader(mrna_path=config['Data_Path']['mrna'],lncrna_path=config['Data_Path']['lncrna'])
    print('-'*50)
    print("Processing")
    mRNA_data,lncRNA_data =load.load()
    df1,df2,df3,df4,df5=processor.preprocess(mRNA_data,lncRNA_data)
    dfs=[df2,df3,df4,df5]

    for i,df in enumerate(dfs):
        # STAGE 01
        print("-"*50)
        print(f"STAGE 01 ,index:{i}")
        Utilil=Utility()
        X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn=Utilil.split_cnn(df)
        X_train_ml, X_val_ml, y_train_ml, y_val_ml=Utilil.split_ml(df)
        print('STAGE 01 Complete')

        #STAGE 02Ingest
        print("-"*50)
        print(f"STAGE 02,index:{i}")
        train=Trainer()
        model_cnn=train.train_cnn(X_train_cnn,y_train_cnn)
        model_svm=train.train_svm(X_train_ml,y_train_ml)
        model_xgboost=train.train_xgboost(X_train_ml,y_train_ml)
        print("STAGE 02 Complete")
        #STAGE 03
        print("-"*50)
        print(f"STAGE 03,index:{i}")
        eval=evaluate()
        eval.evaluater(model_cnn,X_val_cnn,y_val_cnn,'cnn')
        eval.evaluater(model_svm,X_val_ml,y_val_ml,'svm')
        eval.evaluater(model_xgboost,X_val_ml,y_val_ml,'xgboost')
        print("STAGE 03 Complete")
        #STAGE 04
        print("-"*50)
        print(f"STAGE 04,index:{i}")
        plot=Plotter()
        plot.plot_res(model_cnn,X_val_cnn,y_val_cnn,f'cnn_{i}_')
        plot.plot_res(model_svm,X_val_ml,y_val_ml,f'svm_{i}_')
        plot.plot_res(model_xgboost,X_val_ml,y_val_ml,f'xg_{i}_')
        print("STAGE 04 Complete")
        #STAGE 05
        print("-"*50)
        print(f"STAGE 05,index:{i}")
        utils=Utility()
        os.makedirs('trained_models/', exist_ok=True)
        utils.save_model(model_cnn, f'/home/snjay/Desktop/projects/mrna_lncrna_classifier/saved models/model_cnn{i}.h5')
        utils.save_model(model_svm, f'/home/snjay/Desktop/projects/mrna_lncrna_classifier/saved models/model_svm{i}.pkl')
        utils.save_model(model_xgboost, f'/home/snjay/Desktop/projects/mrna_lncrna_classifier/saved models/model_xgboost{i}.json')
        print("STAGE 05 Complete")