###########################################################################
# CUSTOMER LİFETİME VALUE PREDİCTİON ( MÜŞTERİ YAŞAM BOYU DEĞERİ TAHMİNİ)
###########################################################################

# İk olarak bulduğumuz CLTV değerini genel olarak olasıksal bir şekilde inceleyip sonrada bir müşterinin CLTV tahminini yapmaya çalışacağız.

# Önceden hesapladığımız CLTV = (Customer Value / Churn Rate ) * Profit Margin
# Customer Value = Purchase Frequency * Avarage Order Value ( Önceki olan )
#  CLTV = Expected Number of Transaction * Expected Avarage Profit
#  CLTV = BG / NBD Model * Gammma Gamma SubModel

# BG / NBD ( Beta Geometric / Negative Binomal distribution ) ile Expected Number of Transaction

# BG/NBD modeli, Expected Number Of Transaction için iki süreci olasılıksal olarak modeller.
# Transaction Process(buy) + Dropout Process(Till you die )
##############################
# Transaction Process (buy)
##############################

# Alive (Canlı) olduğu sürece, belirli zaman periyodunda, müşteri tarafından gerçekleştirilecek işlem sayısı transaciton rate parametresi ile possion dağılır.

# Bir müşteri alive olduğu sürece kendi transaction rate'i etrafında rastgele satın alma yapmaya devam edecektir.

# Transaction rate'ler her bir müşteriye göre değişir ve tüm kitle için gamma dağılır.(r,a)

############################
# Dropout Process (Till you die)
###########################

# Her bir müşterinin p olasılığı ile dropout rate (dropout probability )'i vardır.
# Bir müşteri alışveriş yaptıktan sonra belirli bir olasılıkla drop olur.
# Dropout rate'ler her bir müşteriye göre değişir ve tüm kitle için beta dağılır (a,b)

#########################################################
# GAMMA GAMMA SUBMODEL
############################################################

# Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.

# Bir müşterinin işlemlerinin parasal değeri (monetary) transaction valuea'larının ortalaması etrafında rastgele dağılır.

# Ortalama transaction value, zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez.

############################################################################
# BG-NBD ve Gamma Gamma ile CLTV Prediction
############################################################################

# 1.Verinin Hazırlanması ( Data Preperation)
# 2.BG-NBD Modeli ile Expected Number of Transaction
# 3.Gamma-Gamma Modeli Expected Avarage Profit
# 4.BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5.CLTV' ye Göre Segmentlerin Oluşturulması
# 6.Çalışmanın fonksiyonlaştırılması

############################################################################
# 1.Verinin Hazırlanması ( Data Preperation)
############################################################################

# Bir e ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.

# Veri seti : RFM/dataset/online_retail_II.xlsx

## Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını inceliyor.

# Değişkenler
#
# InvoiceNo: Faturu numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi.
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün Fiyatı ( Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası.
# Country: Ülke ismi. Müşterilerin yaşadığı ülke.

##########################################
# Gerekli Kütüphane ve Fonksiyonlar
##########################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#################################################
# Verinin Okunması
################################################

df_ = pd.read_excel("RFM/dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()
df.describe().T
df.isnull().sum()
#####################
# Veri Ön İşleme
####################

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

################################################
# LifeTime Veri Yapısının Hazırlanması
###############################################

# recency : Son satın alma üzerinden geçen zaman.Haftalık(kullanıcı özelinde)
# T: Müşterinin yaşı.Haftalık.(analiz tarihinden ne kadar süre önce ilk satın alma yapılmış.)
# frequency : tekrar eden toplam satın alma sayısı
# monetary value : satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate":[lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                                                        lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                         "Invoice" : lambda  Invoice : Invoice.nunique(),
                                         "TotalPrice": lambda  TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

######################################################################
# 2. BG-NBD Modelinin Kurulması
#######################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

########################################################################
# 1 hafta içinde en çok satın alma beklediğiniz 10 müşteri kimdir ?
###########################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purch_1_week"] = bgf.predict(1,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])


########################################################################
# 1 ay içinde en çok satın alma beklediğiniz 10 müşteri kimdir ?
###########################################################################

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purch_1_moth"] = bgf.predict(4,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])
bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()


# Tahmin Sonuçlarının Değerlendirilmesi

plot_period_transactions(bgf)
plt.show()


##########################################################
# 3.GAMMA GAMMA Modelinin Kurulması
###########################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

###########################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin Hesaplanması
##########################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,  #3aylık
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv.head()

cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)

################################################################
# 5. CLTV' ye göre Segmentlerin Oluşturulması
################################################################

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.sort_values(by="clv", ascending=False).head(50)
cltv_final[cltv_final["Segment"] == "A"]
