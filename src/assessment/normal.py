from format import SPLITS, VARS, get_predictions
import fitassesment as fa
import os
import time
import platform
import pandas as pd

cwd = "" # os.getcwd()
thisdir = os.path.join(cwd, 'src/juan/')
imdir = os.path.join(thisdir, 'img/')
resdir = os.path.join(thisdir, 'results/')

def AnalysisOfResiduals(
        experiment, split, variable
    ):
    PREDICTIONS = get_predictions(experiment, split, variable)
    for key, df in PREDICTIONS.items():
        spbleu = df['sp-BLEU']
        models = df.columns[df.columns.get_loc('sp-BLEU')+1:]

        norm = []
        aic = []
        bic = []
        r2 = []
        levene = []
        bartlett = []
        loglikelyhood = []

        evaluation = []

        for model in models:
            directory = os.path.join(
                imdir, *[ a for a in [experiment, split, variable, key, model] ]
            )

            if not os.path.exists(directory):
                os.makedirs(directory)
            
            print(f'Assesment of model\n{experiment} {split} {variable} {key} {model}')

            pred = df[model].to_numpy()
            ds = fa.Errors(pred, spbleu)

            print("Number of samples:", ds.N)

            ds.Graphs(directory, experiment, split, variable, key, model)

            normT = ds.normalityTest()
            aicT = ds.AIC()
            bicT = ds.BIC()
            r2T = ds.R2()
            leveneT = ds.homocedasticityLevene()
            bartlettT = ds.homocedasticityBartlett()
            loglikelyhoodT = ds.logLik

            norm.append(normT)
            aic.append(aicT)
            bic.append(bicT)
            r2.append(r2T)
            levene.append(leveneT)
            bartlett.append(bartlettT)
            loglikelyhood.append(loglikelyhoodT)

            print("\n")

        directory = os.path.join(
            resdir, *[ experiment, split, variable ]
        )

        if not os.path.isdir(directory):
            os.makedirs(directory)

        data = {
            'models': models,
            'Normality p-value': norm,
            'AIC of centered model': aic,
            'BIC of centered model': bic,
            'LogLikelyhood': loglikelyhood,
            'R2 coefficient': r2,
            'Homocedasticity Levene p-value': levene,
            'Homocedasticity bartlett p-value': bartlett
        }

        results_df = pd.DataFrame(data = data)
        results_df.to_csv(os.path.join(directory, key + ".csv"))

def main():
    issues = []

    var = 1

    EXPRS = [
        "1A",
        # "2A",
        # "2B",
        # "2C"
    ]

    for experiment in EXPRS:
        for split in SPLITS[experiment]:
            for variable in VARS[experiment]:
                try: 
                   AnalysisOfResiduals(experiment, split, variable)
                except:
                   print("ERROR running the analysis", experiment, split, variable)
                   issues.append({
                       "experiment": experiment, 
                       "split": split, 
                       "variable": variable
                   })

    for issue in issues:
        print("Issue with", issue["experiment"], issue["split"], issue["variable"])

if __name__ == '__main__':
    main()


    if platform.system() == 'Linux':
        duration = 0.2  # seconds
        freq = 440  # Hz

        for bip in range(4):
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            time.sleep(0.1)
        