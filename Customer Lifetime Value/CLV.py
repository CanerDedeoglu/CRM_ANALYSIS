############################################################
# CUSTOMER LİFETİME VALUE (Müşteri Yaşam Boyu Değeri )
############################################################

# Bir müşterinin bir şirketle kurduğu iişki iletişim sürecinde bu şirkete kazandıracağı parasal değerdir.

# CLTV = ( Customer Value / Churn Rate ) * Profit Margin
# Customer Value = Müşteri Değeri
# Churn Rate = Müşteri terk etme oranı
# Profit Margin = Şirketin belirlediği Kar Oranı

# Customer Value = Avarage Order Value * Purchase Frequency
# Avarage Order Value = Satın alma başına ortalama kazanç.
# Purchase Frequency = Sipariş sıklığı

# Avarage Order Value = Total Price / Total Transaciton

# Purchase Frequency = Total Transaction / Total Customer Of Numbers

# Churn Rate = 1 - Repeat Rate
# Repeat Rate = Birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler
# Profit Margin = Total Price * 0.10

# 1. Veri Hazırlama
# 2. Average Order Value (avarage_value_order = total_price / total_transaction )
# 3. Purchase Frequency ( total_transaction / total_number_of_customers
# 4. Repeat Rate & Churn Rate ( birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5.Profit Margin ( profit_margin = total_price * 0.10)
# 6. Customer Value (customer_value = avarage_order_value * purchase frequency)
# 7. Customer LifeTime Value ( CLTV = (customer_value / churn rate ) * Profit Margin )
# 8. Segmentlerin Oluşturulması
# 9. Tüm işlemlerin fonksiyonlaştırılması


########################################################################
# 1.Veri Hazırlama
#######################################################################

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

import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_excel("RFM/dataset/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

df.head()

df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[(df["Quantity"] > 0)]

df.isnull().sum()
df.dropna(inplace=True)

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby("Customer ID").agg({"Invoice" : lambda x : x.nunique(),
                                        "Quantity": lambda x : x.sum(),
                                        "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

###################################################################################
# 2. Average Order Value (avarage_value_order = total_price / total_transaction )
#####################################################################################

cltv_c.head()

cltv_c["avarage_value_order"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################################################
# 3. Purchase Frequency ( total_transaction / total_number_of_customers
##################################################################################

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

cltv_c.head()

###################################################################################
# 4. Repeat Rate & Churn Rate ( birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
###################################################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

###############################################################################
# 5.Profit Margin ( profit_margin = total_price * 0.10)
################################################################################

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

cltv_c.drop("repeat_rate", axis=1,inplace=True)
cltv_c.drop("prifit_margin", axis=1,inplace=True)

##############################################################################
# 6. Customer Value (customer_value = avarage_order_value * purchase frequency)
##############################################################################

cltv_c["customer_value"] = cltv_c["avarage_value_order"] * cltv_c["purchase_frequency"]

#############################################################################
# 7. Customer LifeTime Value ( CLTV = (customer_value / churn rate ) * Profit Margin )
#############################################################################
cltv_c["cltv"] = ( cltv_c["customer_value"] / churn_rate ) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv",ascending=False).head()

##############################################################################
# 8. Segmentlerin Oluşturulması
##############################################################################

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_c.groupby("segment").agg( {"count", "mean", "sum"})

cltv_c.to_csv("cltv.csv")

##################################################################################
# 9. Tüm işlemlerin fonksiyonlaştırılması
##################################################################################

def create_cltv_c(dataframe, profit=0.10):

    #Veriyi hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Price"] * dataframe["Quantity"]

    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                            "Quantity": lambda x: x.sum(),
                                            "TotalPrice": lambda x: x.sum()})

    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # avg_order_value
    cltv_c["avarage_value_order"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    #purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    #repeat rate & churn rate

    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

    churn_rate = 1 - repeat_rate

    # profit margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * profit

    # customer value
    cltv_c["customer_value"] = cltv_c["avarage_value_order"] * cltv_c["purchase_frequency"]

    # customer lifeTime Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

    # segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

df = df_.copy()

clv = create_cltv_c(df)