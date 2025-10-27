from validphys.api import API

top_data = [
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_PTT'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_PTT-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_YT'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_YT-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT'}, # bugged covariance
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT-NORM'}, # bugged covariance
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT'},
    {'dataset': 'ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT-NORM'},
    {'dataset': 'ATLAS_TTBAR_13TEV_TOT_X-SEC'},
    {'dataset': 'ATLAS_TTBAR_7TEV_TOT_X-SEC'},
    {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR'},
    {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_PTT'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_PTT-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM'},
    {'dataset': 'ATLAS_TTBAR_8TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_PTT'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_PTT-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_MTTBAR'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_MTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YTTBAR'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_2016_DIF_YT', 'variant': 'legacy'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_MTTBAR'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_YTTBAR'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_PTT'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_PTT-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_YT'},
    {'dataset': 'CMS_TTBAR_13TEV_LJ_DIF_YT-NORM'},
    {'dataset': 'CMS_TTBAR_13TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_5TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_7TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_PTT-YT-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM', 'variant': 'legacy_data'}, # low chi2, see issue #2356
    {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_PTT-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YT-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_MTTBAR-NORM'},
    {'dataset': 'CMS_TTBAR_8TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_138FB-1_DIF_MTTBAR'},  # high chi2 due to correlations
    {'dataset': 'CMS_TTBAR_13TEV_2L_138FB-1_DIF_PTT'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_138FB-1_DIF_YT'},
    {'dataset': 'CMS_TTBAR_13TEV_2L_138FB-1_DIF_MTTBAR-YTTBAR'},
    {'dataset': 'ATLAS_TTBAR_13P6TEV_TOT_X-SEC'},
    {'dataset': 'ATLAS_TTBAR_5TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_13P6TEV_TOT_X-SEC'},
    {'dataset': 'CMS_TTBAR_13TEV_35P9FB-1_TOT_X-SEC'},
]

def compute_z(i):
    dist = i['dataset']
    z = API.covmat_stability_characteristic(dataset_input=i, theoryid=40_009_000, use_cuts="internal")
    print(f'{dist}: {z}')

for i in top_data:
    compute_z(i)
