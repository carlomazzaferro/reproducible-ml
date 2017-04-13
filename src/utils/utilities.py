import os
import pandas
import glob
import json


def clean_data(data_file):

    # clean

    return data_file


def save_results_to_file(params, df, plot, fpr, tpr):

    if params['data_dir'][-1] == '/':
        final_res_dir = params['data_dir'] + params['run_id']
    else:
        final_res_dir = params['data_dir'] + '/' + params['run_id']

    os.makedirs(final_res_dir)

    with open(final_res_dir + '/params_dump.json', 'w') as p:
        json.dump(params, p)

    df.to_csv(final_res_dir + '/preds_targs.csv')
    plot.savefig(final_res_dir + '/roc_curve.png', orientation='portrait', bbox_inches='tight')
    fp_tp_df = pandas.DataFrame([fpr, tpr], index=['FP', 'TP']).T
    fp_tp_df.to_csv(final_res_dir + '/FP_TP.csv')

    print('Results written to %s' % final_res_dir)


def assign_id(params, res_dir):

    res_files = glob.glob(res_dir + '%s_%s*' % (params['model_type'], str(params['DRUG_ID'])))
    num_files = len(res_files)
    return params['model_type'] + '_' + str(params['DRUG_ID']) + '_' + str(num_files)

