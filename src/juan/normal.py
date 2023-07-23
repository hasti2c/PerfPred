from format import EXPRS, SPLITS, VARS, get_predictions
import fitassesment as fa
import os
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

        for model in models:
            directory = os.path.join(
                imdir, 
                *[
                    a for a in [experiment, split, variable, key,
                    model]
                ]
            )

            if not os.path.exists(directory):
                os.makedirs(directory)
            

            print(f'\n\n Assesment of {model} model in table {experiment}-{split}-{variable}-{key}-{model}\n')

            pred = df[model].to_numpy()
            ds = fa.Errors(pred, spbleu)

            ds.varianceEvolution(
                f'Variance Evolution in errors: {experiment}-{split}-{variable}-{key}-{model}',
                filename = f'{experiment}-{split}-{variable}-{key}-{model}-variance',
                path = directory
            )
            ds.pdf_comparison(
                f'Comparision of density functions: {experiment}-{split}-{variable}-{key}-{model}',
                filename = f'{experiment}-{split}-{variable}-{key}-{model}-densities',
                path = directory
            )
            ds.QQ(
                f'QQ-plot for non centered model: {experiment}-{split}-{variable}-{key}-{model}',
                filename = f'{experiment}-{split}-{variable}-{key}-{model}-QQ',
                path = directory
            )
            ds.QQcent(
                f'QQ-plot for centered model: {experiment}-{split}-{variable}-{key}-{model}',
                filename = f'{experiment}-{split}-{variable}-{key}-{model}-QQcent',
                path = directory
            )

            norm.append(ds.normalityTest())
            print(
                "Normality test p-value:", ds.normalityTest()
            )
                # print("ERROR in Normality test")

            try:
                aic.append(ds.AICcent())
                print(
                    "AIC centered model:", ds.AICcent()
                )
            except:
                print("ERROR in AIC test")
            
            try:
                bic.append(ds.BICcent())
                print(
                    "BIC centered model:", ds.BICcent()
                )
            except:
                print("ERROR in BIC test")

            try:
                r2.append(ds.R2())
                print(
                    "R2:", ds.R2()
                )
            except:
                print("ERROR in R2")
            
            levene.append(ds.homocedasticityLevene())
            print(
                "Homocedasticity Levene p-value:", ds.homocedasticityLevene()
            )
                # print("ERROR in Levene")

            bartlett.append(ds.homocedasticityBartlett())
            print(
                "Homocedasticity Bartlett p-value:", ds.homocedasticityBartlett(),
            )
                # print("ERROR in Bartlett")

        directory = os.path.join(
            resdir, 
            *[
                experiment, split, variable
            ]
        )
        if not os.path.isdir(directory):
            os.makedirs(directory)

        data = {
            'models': models,
            'Normality p-value': norm,
            'AIC of centered model': aic,
            'BIC of centered model': bic,
            'R2 coefficient': r2,
            'Homocedasticity Levene p-value': levene,
            'Homocedasticity bartlett p-value': bartlett,
        }

        results_df = pd.DataFrame(data = data)
        results_df.to_csv(os.path.join(directory, key + ".csv"))

def main():
    issues = []

    experiment = "2C"
    split = "lang"
    variable = VARS[experiment][14]

    PREDS = get_predictions(experiment, split, variable)

    # for experiment in EXPRS:
    #     for split in SPLITS[experiment]:
    #         for variable in VARS[experiment]:
    #             try: 
    #                 PREDS = get_predictions(experiment, split, variable)
    #                 # for key in PREDS:
    #                 #     print("Num observations in", experiment, split, variable)
    #                 #     print(len(PREDS[key]))
    #                 # # AnalysisOfResiduals(experiment, split, variable)
    #             except:
    #                print("ERROR running the analysis", experiment, split, variable)
    #                issues.append({
    #                    "experiment": experiment, 
    #                    "split": split, 
    #                    "variable": variable
    #                })

    for issue in issues:
        print("Issue with", issue["experiment"], issue["split"], issue["variable"])

if __name__ == '__main__':
    main()

    duration = 5  # milliseconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))