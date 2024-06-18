import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re


# web interface part
st.title("Spam Email Classification")
st.write("""### Please input the content of your email""")
user_input = st.text_input("Your email:")
ok = st.button("That's it!")

# all features' names
my_list = ['word_freq_make',
 'word_freq_address',
 'word_freq_all',
 'word_freq_3d',
 'word_freq_our',
 'word_freq_over',
 'word_freq_remove',
 'word_freq_internet',
 'word_freq_order',
 'word_freq_mail',
 'word_freq_receive',
 'word_freq_will',
 'word_freq_people',
 'word_freq_report',
 'word_freq_addresses',
 'word_freq_free',
 'word_freq_business',
 'word_freq_email',
 'word_freq_you',
 'word_freq_credit',
 'word_freq_your',
 'word_freq_font',
 'word_freq_000',
 'word_freq_money',
 'word_freq_hp',
 'word_freq_hpl',
 'word_freq_george',
 'word_freq_650',
 'word_freq_lab',
 'word_freq_labs',
 'word_freq_telnet',
 'word_freq_857',
 'word_freq_data',
 'word_freq_415',
 'word_freq_85',
 'word_freq_technology',
 'word_freq_1999',
 'word_freq_parts',
 'word_freq_pm',
 'word_freq_direct',
 'word_freq_cs',
 'word_freq_meeting',
 'word_freq_original',
 'word_freq_project',
 'word_freq_re',
 'word_freq_edu',
 'word_freq_table',
 'word_freq_conference',
 'char_freq_;',
 'char_freq_(:           ',
 'char_freq_[:            ',
 'char_freq_!:            ',
 'char_freq_$:          ',
 'char_freq_#:         ',
 'capital_run_length_average: ',
 'capital_run_length_longest:',
 'capital_run_length_total: ']
# I changed 'char_freq_[:            ' to 'char_freq_' during xgboost training
col_list = ['word_freq_make',

 'word_freq_address',
 'word_freq_all',
 'word_freq_3d',
 'word_freq_our',
 'word_freq_over',
 'word_freq_remove',
 'word_freq_internet',
 'word_freq_order',
 'word_freq_mail',
 'word_freq_receive',
 'word_freq_will',
 'word_freq_people',
 'word_freq_report',
 'word_freq_addresses',
 'word_freq_free',
 'word_freq_business',
 'word_freq_email',
 'word_freq_you',
 'word_freq_credit',
 'word_freq_your',
 'word_freq_font',
 'word_freq_000',
 'word_freq_money',
 'word_freq_hp',
 'word_freq_hpl',
 'word_freq_george',
 'word_freq_650',
 'word_freq_lab',
 'word_freq_labs',
 'word_freq_telnet',
 'word_freq_857',
 'word_freq_data',
 'word_freq_415',
 'word_freq_85',
 'word_freq_technology',
 'word_freq_1999',
 'word_freq_parts',
 'word_freq_pm',
 'word_freq_direct',
 'word_freq_cs',
 'word_freq_meeting',
 'word_freq_original',
 'word_freq_project',
 'word_freq_re',
 'word_freq_edu',
 'word_freq_table',
 'word_freq_conference',
 'char_freq_;',
 'char_freq_(:           ',
 'char_freq_',
 'char_freq_!:            ',
 'char_freq_$:          ',
 'char_freq_#:         ',
 'capital_run_length_average: ',
 'capital_run_length_longest:',
 'capital_run_length_total: ']

 # get the target string
str_list = []
for i in range(54):
    str_list.append(my_list[i][10:])

my_string = user_input
if(len(my_string) == 0):
          st.warning('Please input email')
else:
          my_data = []
          # 前面那些算frequncy的先按顺序放进去
          for i in range(54):
                    my_data.append(100 * my_string.count(str_list[i]) / len(my_string) )

          # get all uninterrupted sequences of capital letters in a list of strings
          string_len = re.findall('[A-Z]+[A-Z]+[A-Z]*', my_string)

          # average length of uninterrupted sequences of capital letters
          if len(string_len) != 0:
                    my_data.append(sum( map(len, string_len) ) / len(string_len))
                    # length of longest uninterrupted sequence of capital letters
                    my_data.append(len(max(string_len, key = len)))
                    # sum of length of uninterrupted sequences of capital letters
                    my_data.append(sum( map(len, string_len) ))
          else:
                    my_data.append(0)
                    my_data.append(0)
                    my_data.append(0)

          loaded_model = pickle.load(open('/Users/david/Desktop/IntroDS/project/webapp/model.pkl', 'rb'))
          res = dict(zip(col_list, my_data))
          dft = pd.DataFrame([res])

          cols_when_model_builds = loaded_model.get_booster().feature_names
          dft = dft[cols_when_model_builds]

          if ok:
                    result = loaded_model.predict(dft)[0]
                    if result == 0:
                              st.subheader(f"Congrats! This is a HAM!")
                    else:
                              st.subheader(f"Oops! This looks like a SPAM!")