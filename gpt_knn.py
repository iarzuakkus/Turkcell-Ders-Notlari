from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 📡 Veritabanı bağlantısı
engine = create_engine("postgresql://postgres:12345@localhost:5432/gyk1")

# 📥 Veriyi çek
df = pd.read_sql(
    """
    SELECT o.customer_id, o.order_date, p.category_id, p.product_id, od.unit_price, od.quantity, od.discount
    FROM orders AS o
    INNER JOIN order_details AS od ON o.order_id = od.order_id
    INNER JOIN products AS p ON p.product_id = od.product_id
    """, 
    con=engine
)

# 🕐 Zaman sütunları
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_month'] = df['order_date'].dt.month
df['order_day'] = df['order_date'].dt.day

# 🔄 customer_id encode
df['customer_id'] = LabelEncoder().fit_transform(df['customer_id'].astype(str))

# 🧠 🔥 En çok geçen ilk 20 product_id dışındakileri 'Other' olarak etiketle
top_products = df['product_id'].value_counts().nlargest(20).index
df['product_id_clean'] = df['product_id'].where(df['product_id'].isin(top_products), 'Other')

# 🔢 product_id_clean için get_dummies (sütun patlamadan)
df = pd.get_dummies(df, columns=['product_id_clean'], drop_first=True)

# 🎯 Özellikler ve hedef
X = df.drop(columns=['quantity', 'order_date', 'product_id'])  # 'product_id' artık yerine 'product_id_clean' var
y = df['quantity']

# 🔀 Eğitim / test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Decision Tree Regressor (dengeli ayarlarla)
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 🔮 Tahmin ve değerlendirme
y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R² Skoru:", r2)

# Opsiyonel: İlk birkaç satırı yazdır
print(df.head())
