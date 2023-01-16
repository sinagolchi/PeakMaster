import pickle
import streamlit as st
import re
import pandas as pd
from io import StringIO
from core import sample
from core import predict
import matplotlib.pyplot as plt

#%%

st.title('LC-MS/MS Peak Table Processing')
with st.sidebar:
    dilution_file = st.file_uploader('Dilution factors')

# with st.sidebar:
#     demo = st.checkbox('Demo mode',value=True)

demo = False

calib_tab, sample_tab = st.tabs(['Calibration','Analysis'])

file = None
with calib_tab:
    if not demo:
        file = st.file_uploader('Upload calibration peak table',help='ASCII export from LabSolutions')
        calib_file = st.file_uploader('Upload .cal file')
    else:
        file = open('Calibration Standards Data.txt','r')


@st.experimental_singleton
def process_file(file):

    samples = []
    dict = {}
    if demo:
        pattern = r'(?<=Name,)(.*?)(?=\n)'
        pattern_b = r'(?<=,Average,)(.*?)(?=\n)'
    else:
        file = StringIO(file.getvalue().decode("utf-8"))
        pattern = r'(?<=Name,)(.*?)(?=\r)'
        pattern_b = r'(?<=,Average,)(.*?)(?=\r)'
    compound = None


    for line in file.readlines():

        samples.append(line)


        a = re.search(pattern,str(line))
        if a is not None:
            compound = a[0]

        b = re.search(pattern_b,str(line))
        if b is not None:
            dict.update({compound:samples})
            samples = []

    #%%
    for compound, data_list in zip(dict.keys(), dict.values()):
        if compound == 'PFBA':
            del data_list[-1]
        else:
            del data_list[0:5]
            del data_list[-1]

    for compound, data_list in zip(dict.keys(), dict.values()):
        del data_list[0:2]
    #%%
    df_dict = {}
    for compound , data_list in zip(dict.keys(), dict.values()):
        df = pd.DataFrame([sub.split(",") for sub in data_list])
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.set_index('')
        df['Data Filename'] = df['Data Filename'].apply(lambda s: s.removesuffix('.lcd'))
        df = df.set_index('Data Filename',drop=True)
        df_dict.update({compound:df})

    #%%
    dict_formatted = {}
    dict_formatted.update({'Points':list(df_dict['PFBA'].index)})

    for compound in df_dict.keys():
        areas = []
        for point in df_dict['PFBA'].index:
            areas.append(df_dict[compound].loc[point,'Area'])
        dict_formatted.update({compound:areas})

    df_formatted = pd.DataFrame(dict_formatted)

    df_formatted.set_index('Points',inplace=True)
    # with sample_tab:
    #     st.dataframe(df_formatted)
    df = df_formatted[df_formatted.columns[list(~df_formatted.columns.str.contains('qualifier',case=False))]]
    # with sample_tab:
    #     st.dataframe(df)

    #st.dataframe(df)




    return df

ISTD_dict = {'PFBA':'MPFBA','PFPeA':'MPFPeA','PFBS':'MPFBS','42FTS':'M42FTS','PFHxA':'MPFHxA','PFHpA':'MPFHpA','PFHxS':'MPFHxS','62FTS':'M62FTS','PFOA':'MPFOA','PFNA':'MPFNA','PFOS':'MPFOS','82FTS':'M82FTS','PFDA':'MPFDA','MeFOSAA':'M-MeFOSAA','FOSA':'M-FOSA','EtFOSAA':'M-EtFOSAA','PFUdA':'MPFUdA','PFDoA':'MPFDoA','PFTeDA':'MPFTeDA'}

def ISTD_map(data,map,use_name=True):
    df = pd.DataFrame()
    if use_name:
        for c_key in map.keys():
            df[c_key] = data[c_key]/data[map[c_key]]
    else:
        for c_key in map.keys():
            df[c_key+'/'+map[c_key]] = data[c_key]/data[map[c_key]]
    return df





def process_calib(df):
    df = df.replace('-----',0)
    df = df.astype(float)
    df_concs = pd.read_csv('Calib concs.csv', index_col='Points')
    #%%
    from core import sample
    ISTD_dict = {'PFBA': 'MPFBA', 'PFPeA': 'MPFPeA', 'PFBS': 'MPFBS', '42FTS': 'M42FTS', 'PFHxA': 'MPFHxA',
                 'PFHpA': 'MPFHpA', 'PFHxS': 'MPFHxS', '62FTS': 'M62FTS', 'PFOA': 'MPFOA', 'PFNA': 'MPFNA',
                 'PFOS': 'MPFOS', '82FTS': 'M82FTS', 'PFDA': 'MPFDA', 'MeFOSAA': 'M-MeFOSAA', 'FOSA': 'M-FOSA',
                 'EtFOSAA': 'M-EtFOSAA', 'PFUdA': 'MPFUdA', 'PFDoA': 'MPFDoA', 'PFTeDA': 'MPFTeDA'}
    calib_set = []

    for compound in ISTD_dict.keys():
        samples = []
        for area,ISTD_area, conc, name in zip(df[compound],df[ISTD_dict[compound]],df_concs[compound],df.index):
            samples.append(sample(name,compound,ISTD_dict[compound],conc,area,ISTD_area))
        calib_set.append(samples)

    return calib_set



def load_calibration(uploaded_file):
    return pickle.load(uploaded_file)

def review_calib(calibset):
    with calib_tab:
        intercept = st.checkbox('Do not fit Intercept (Force through zero)')
        outliers = st.multiselect('Outliers', options=[c.name for c in calibset[0]])

    calib_set = [[c for c in lst if c.name not in outliers] for lst in calibset]
    from core import calibration

    calibrated = {}
    for calib in calib_set:
        calibrated.update({calib[0].comp: calibration(calib, True, split=False, split_point=500, intercept=not intercept)})
    return calibrated

if file is not None:
    calib_df = process_file(file)
    calib_set = process_calib(calib_df)
    calibrated = review_calib(calib_set)

elif calib_file is not None:
    calibrated = load_calibration(calib_file)
else:
    st.stop()





#%%

with calib_tab:
    comp_select = st.selectbox('select compound for preview',options=[c for c in calibrated.keys()])
    st.pyplot(calibrated[comp_select].plot())

    st.download_button('Download calibration file', data=pickle.dumps(calibrated), file_name='calib.cal')


#%%

with sample_tab:
    batch_file = st.file_uploader('Upload sample peak table')

def process_sample(df,calib):

    #%%
    df_concs = pd.read_csv('Calib concs.csv', index_col='Points')

    df = df.replace('-----',None)
    df = df.astype(float)
    sample_set = {}

    for compound in ISTD_dict.keys():
        samples = []
        for area,ISTD_area, name in zip(df[compound],df[ISTD_dict[compound]],df.index):
            s = sample(name,compound,ISTD_dict[compound],None,area,ISTD_area)
            predict(s, calib)
            samples.append(s)
        sample_set.update({compound:samples})

    return sample_set

def make_sample_df(sample_set):

    dict_for_df = {}
    for compound in sample_set.keys():
        concs = []
        for point in sample_set[compound]:
            concs.append(point.conc)
        dict_for_df.update({compound:concs})


    dict_for_df.update({'Points':[p.name for p in list(sample_set.values())[0]]})

    sample_df = pd.DataFrame(dict_for_df)
    sample_df.set_index('Points',inplace=True)
    return sample_df

def apply_dilution_factor(df,df_dilute):
    df*df_dilute
    return df


if batch_file is not None:
    samples_set = process_file(batch_file)
    p_samples = process_sample(samples_set,calibrated)
    df_concs = make_sample_df(p_samples)
    with sample_tab:
        st.header('Raw concentrations without correction')
        st.dataframe(df_concs)
    df_concs2 = df_concs*pd.read_csv(dilution_file,index_col='Points')
    with sample_tab:
        st.header('With dilution factor correction')
        st.dataframe(df_concs2)
    st.download_button('Download results',data=df_concs2.to_csv(),file_name='Concs.csv')
else:
    st.stop()


