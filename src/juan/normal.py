from format import PREDICTIONS, langs
import fitassesment as fa
import os
import pandas as pd

thisdir = 'src/juan/'
imdir = thisdir + 'img/'
resdir = thisdir + 'results/'

dirs = [
    thisdir,
    imdir,
    resdir
]

for d in dirs:
    if not os.path.isdir(d):
        os.mkdir(d)

# l = langs[0]
for l in langs:
    df = PREDICTIONS[l]

    models = [
        m for m in df.columns[6:17]
    ]
    sp_bleu = df['sp-BLEU'].to_numpy()

    if not os.path.isdir(imdir + l):
        os.mkdir(imdir + l)

    norm = []
    aic = []
    bic = []
    r2 = []
    levene = []
    bartlett = []
    
    for model in models:
        directory = imdir + l + "/" + model + "/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        print(f'Assesment of {model} model in language {l}')

        pred = df[model].to_numpy()
        ds = fa.Errors(pred, sp_bleu)

        ds.varianceEvolution(
            f'Variance Evolution in errors: {model}-{l}',
            filename = f'{model}-{l}-variance',
            path = directory
        )
        ds.pdf_comparison(
            f'Comparision of density functions: {model}-{l}',
            filename = f'{model}-{l}-densities',
            path = directory
        )
        ds.QQ(
            f'QQ-plot for non centered model: {model}-{l}',
            filename = f'{model}-{l}-QQ',
            path = directory
        )
        ds.QQcent(
            f'QQ-plot for centered model: {model}-{l}',
            filename = f'{model}-{l}-QQcent',
            path = directory
        )

        print(
            "Normality test p-value:", ds.normalityTest()
        )
        norm.append(ds.normalityTest())
        print(
            "AIC centered model:", ds.AICcent()
        )
        aic.append(ds.AICcent())
        print(
            "BIC centered model:", ds.BICcent(),
            end = "\n\n"
        )
        bic.append(ds.BICcent())
        print(
            "R2:", ds.R2(),
            end = "\n\n"
        )
        r2.append(ds.R2())
        print(
            "Homocedasticity Levene p-value:", ds.homocedasticityLevene(),
            end = "\n\n"
        )
        levene.append(ds.homocedasticityLevene())
        print(
            "Homocedasticity Bartlett p-value:", ds.homocedasticityLevene(),
            end = "\n\n"
        )
        bartlett.append(ds.homocedasticityBartlett())

        # print(
        #     "homocedasticity Levene test p-value:", ds.homocedasticityLevene(),
        #     end = "\n\n"
        # )
        # print(
        #     "homocedasticity Bartlett test p-value:", ds.homocedasticityBartlett(),
        #     end = "\n\n"
        # )

    data = {
        'models': models,
        'Normality p-value': norm,
        'AIC of centered model': aic,
        'BIC of centered model': bic,
        'R2 coefficient': r2,
        'Homocedasticity Levene p-value': levene,
        'Homocedasticity bartlett p-value': bartlett,
    }

    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    results_df = pd.DataFrame(data = data)
    results_df.to_csv(resdir + l + ".csv")