#import module yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

#buka file dataset
url = 'Household energy bill data.csv'
df = pd.read_csv(url)

#tampilkan contoh isi dataset
print("=".rjust(117,'='))
print("Data Example".center(117,' '))
print("=".rjust(117,'='))
print(df)

#cek dataset apakah perlu dibersihkan dulu
print("=".rjust(50,'='))
print("Data Info".center(50,' '))
print("=".rjust(50,'='))
print(df.info())
print("=".rjust(78,'='))

#buat prediksi dengan semua variabel yang ada
lm = smf.ols('amount_paid ~ num_rooms + num_people + housearea + is_ac + is_tv + is_flat + ave_monthly_income + num_children + is_urban',
             data = df).fit()
print(lm.summary())
print("\n")

#buat prediksi hanya dengan variable yang significant
print("=".rjust(78,'='))
lm1 = smf.ols('amount_paid ~ num_people + housearea + is_ac + is_tv + is_flat + ave_monthly_income + num_children + is_urban',
             data = df).fit()
print(lm1.summary())

#tampilkan residual plot lengkap dengan garis horisontal di nol
plt.subplot(1, 2, 1)
plt.scatter(lm1.fittedvalues, lm1.resid, marker=".", c = "k")
plt.ylabel("residual")
plt.xlabel("fitted values")
plt.axhline([0])

#tampilkan distribusi error
plt.subplot(1, 2, 2)
plt.hist(lm1.resid, color='tab:blue', alpha=0.4)
plt.xlabel("residual")
plt.ylabel("count")

plt.show()
