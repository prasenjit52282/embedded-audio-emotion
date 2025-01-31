import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(rp):
    rp['ts']=rp['Time (ms)'].apply(int)
    return rp.groupby('ts')[['USB Avg Power (mW)','USB Avg Current (mA)']].mean()

plt.rcParams.update({
    'figure.figsize': (6,4),        # Set default figure size (width, height) in inches
    'axes.labelsize': 18,            # Font size for x and y labels
    'xtick.labelsize': 18,           # Font size for x-axis ticks
    'ytick.labelsize': 19,           # Font size for y-axis ticks
    'axes.labelweight': 'bold'
})

df=pd.read_csv("extracted_acoustic_features.csv")
df['Emotion']=df['Emotion'].apply(str.capitalize)

#Distribution of Zero Crossing Rate Values Across Emotions
df['Zero Crossing Rate'] = df['Zero Crossing Rate'].astype(float)
emotions_of_interest = ['Neutral', 'Angry', 'Fear', 'Surprise']
filtered_df = df[df['Emotion'].isin(emotions_of_interest)]
plt.figure()
sns.boxplot(x='Emotion', y='Zero Crossing Rate', data=filtered_df, showfliers=False, width=0.6)
plt.xlabel('Emotions') 
plt.ylabel('Zero Crossing Rate') 

plt.tight_layout()
plt.savefig("./logs/bp1.pdf")


#Distribution of Energy Values Across Emotions
df['Energy'] = df['Energy'].astype(float)
emotions_of_interest = ['Happy', 'Angry', 'Sad', 'Disgust']
filtered_df = df[df['Emotion'].isin(emotions_of_interest)]
plt.figure()
sns.boxplot(x='Emotion', y='Energy', data=filtered_df ,showfliers=False, width=0.6)
plt.xlabel('Emotions')
plt.ylabel('Energy')

plt.tight_layout()
plt.savefig("./logs/bp2.pdf")


#Distribution of Shimmer Values Across Emotions
df['Shimmer'] = df['Shimmer'].astype(float)
emotions_of_interest = ['Happy', 'Angry', 'Fear', 'Disgust']
filtered_df = df[df['Emotion'].isin(emotions_of_interest)]
plt.figure()
sns.boxplot(x='Emotion', y='Shimmer', data=filtered_df,showfliers=False,width=0.6)
plt.xlabel('Emotions')
plt.ylabel('Shimmer')

plt.tight_layout()
plt.savefig("./logs/bp3.pdf")


#Distribution of MFCC_1 Values Across Emotions
df['MFCC_1'] = df['MFCC_1'].astype(float)
emotions_of_interest = ['Neutral', 'Disgust', 'Fear', 'Surprise']
filtered_df = df[df['Emotion'].isin(emotions_of_interest)]
plt.figure()
sns.boxplot(x='Emotion', y='MFCC_1', data=filtered_df,showfliers=False,width=0.6)

plt.xlabel('Emotions')
plt.ylabel('MFCC$_{\mathbf{1}}$')

plt.tight_layout()
plt.savefig("./logs/bp4.pdf")



#power profile
rp_r=pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679384517895600-rp2040.csv").iloc[0:50000]
esp32_r=pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679385979375202-esp32.csv").iloc[0:50000]
s3_r=pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679393896273357_esp32s3.csv").iloc[0:50000]
nrf_r=pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679388785892176_nrf52.csv").iloc[0:50000]

rp=preprocess(pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679384517895600-rp2040.csv").iloc[0:50000])
esp32=preprocess(pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679385979375202-esp32.csv").iloc[0:50000])
s3=preprocess(pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679393896273357_esp32s3.csv").iloc[0:50000])
nrf=preprocess(pd.read_csv("./logs/power_profiles/PowerBox_23573_-8584679388785892176_nrf52.csv").iloc[0:50000])

#power draw
plt.figure(figsize=(6,4))
sns.boxplot([d['USB Avg Power (mW)'].values for d in [esp32_r,s3_r,nrf_r,rp_r]],width=0.6)
plt.xticks([0,1,2,3],labels=['ESP32','ESP32S3','nRF52','RP2040'],fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Power (mW)',fontdict=dict(size=18),weight="bold")
plt.xlabel('Devices',fontdict=dict(size=18),weight="bold")
plt.ylim(0,300)
plt.tight_layout()
plt.savefig("./logs/power_draw.pdf")


#current draw
plt.figure(figsize=(6,4))

l=['ESP32','ESP32S3','nRF52','RP2040']
for i,d in enumerate([esp32,s3,nrf,rp]):
    plt.plot(d['USB Avg Current (mA)'].values,label=l[i],lw=2)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylim(0,100)
plt.ylabel('Current (mA)',fontdict=dict(size=18,weight="bold"))
plt.xlabel('Time (ms)',fontdict=dict(size=18),weight="bold")
plt.legend(fontsize=15,frameon=False,loc='upper left')
plt.annotate('Prediction', xy=(4500, 60), xytext=(5500, 70),
            arrowprops=dict(facecolor='black', shrink=0.05),fontsize=16)
plt.tight_layout()
plt.savefig("./logs/current_draw.pdf")