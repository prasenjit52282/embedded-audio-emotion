#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[43]:


def preprocess(rp):
    rp['ts']=rp['Time (ms)'].apply(int)
    return rp.groupby('ts')[['USB Avg Power (mW)','USB Avg Current (mA)']].mean()


# In[47]:


rp_r=pd.read_csv("./powe_wip/PowerBox_23573_-8584679384517895600-rp2040.csv").iloc[0:50000]
esp32_r=pd.read_csv("./powe_wip/PowerBox_23573_-8584679385979375202-esp32.csv").iloc[0:50000]
s3_r=pd.read_csv("./powe_wip/PowerBox_23573_-8584679393896273357_esp32s3.csv").iloc[0:50000]
nrf_r=pd.read_csv("./powe_wip/PowerBox_23573_-8584679388785892176_nrf52.csv").iloc[0:50000]


# In[48]:


rp=preprocess(pd.read_csv("./powe_wip/PowerBox_23573_-8584679384517895600-rp2040.csv").iloc[0:50000])
esp32=preprocess(pd.read_csv("./powe_wip/PowerBox_23573_-8584679385979375202-esp32.csv").iloc[0:50000])
s3=preprocess(pd.read_csv("./powe_wip/PowerBox_23573_-8584679393896273357_esp32s3.csv").iloc[0:50000])
nrf=preprocess(pd.read_csv("./powe_wip/PowerBox_23573_-8584679388785892176_nrf52.csv").iloc[0:50000])


# In[117]:


plt.figure(figsize=(6,4))
sns.boxplot([d['USB Avg Power (mW)'].values for d in [esp32_r,s3_r,nrf_r,rp_r]],width=0.5)
plt.xticks([0,1,2,3],labels=['ESP-WR32','ESP32S3','nRF52840','RP2040'],fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Power (mW)',fontdict=dict(size=16))
plt.xlabel('Devices',fontdict=dict(size=16))
plt.ylim(0,300)
plt.tight_layout()
plt.savefig("power_draw.pdf")


# In[118]:


plt.figure(figsize=(6,4))

l=['ESP-WR32','ESP32S3','nRF52840','RP2040']
for i,d in enumerate([esp32,s3,nrf,rp]):
    plt.plot(d['USB Avg Current (mA)'].values,label=l[i],lw=2)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylim(0,100)
plt.ylabel('Current (mA)',fontdict=dict(size=16))
plt.xlabel('Time (ms)',fontdict=dict(size=16))
plt.legend(fontsize=15,frameon=False,loc='upper left')
plt.annotate('Prediction', xy=(4500, 60), xytext=(5500, 70),
            arrowprops=dict(facecolor='black', shrink=0.05),fontsize=16)
plt.tight_layout()
plt.savefig("current_draw.pdf")

