from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# 📡 Veritabanına bağlan
engine = create_engine("postgresql://postgres:12345@localhost:5432/gyk1")

# 🧾 Verileri SQL'den çek
df = pd.read_sql(
    """
    SELECT o.customer_id, o.order_date, p.category_id, p.product_id, od.unit_price, od.quantity, od.discount
    FROM orders AS o
    INNER JOIN order_details AS od ON o.order_id = od.order_id
    INNER JOIN products AS p ON p.product_id = od.product_id
    """, 
    con=engine
)

# 📅 order_date'i datetime formatına çevir
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_month'] = df['order_date'].dt.month
df['order_day'] = df['order_date'].dt.day

# 💸 Toplam harcama sütunu (isteğe bağlı ama faydalı olabilir)
df['total_spent'] = df['unit_price'] * df['quantity'] * (1 - df['discount'])

# 🔄 product_id ve customer_id'yi kategorik olarak işaretle
df['product_id'] = df['product_id'].astype(str)
df['customer_id'] = df['customer_id'].astype(str)

# 🔍 One-Hot Encoding (sıralı gibi algılanmasın diye)
df = pd.get_dummies(df, columns=['product_id', 'customer_id'], drop_first=True)

# 🎯 Hedef ve bağımsız değişkenleri ayır
X = df.drop(columns=['quantity', 'order_date'])
y = df['quantity']

# 🔀 Train-Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 🔮 Tahmin ve metrikler
y_pred = model.predict(X_test)

print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
print(df)