from sqlalchemy import create_engine, MetaData
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score, root_mean_squared_error, mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:12345@localhost:5432/gyk1")

import pandas as pd


df = pd.read_sql(
    """
    SELECT o.customer_id, o.order_date, p.category_id, p.product_id, od.unit_price, od.quantity, od.discount
    FROM orders AS o
    FULL OUTER JOIN order_details AS od ON o.order_id = od.order_id
    FULL OUTER JOIN products AS p ON p.product_id = od.product_id
    """, 
    con=engine
)
print(df.info())

df['total_spent'] = df.unit_price * df.quantity * (1- df.discount)

# Tarihi ay bazına çevir:
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')

# Aylık satış özeti:
monthly_sales = df.groupby('month')['quantity'].sum().reset_index()

# Ürün bazlı satış:
product_sales = df.groupby('product_id')[['quantity','total_spent']].sum().reset_index()

#musteri bazli satis
customer_sales = df.groupby('customer_id').size().reset_index(name='order_counts')

# Müşteri segmentasyonu (örnek kural)
customer_sales['Segment'] = pd.cut(customer_sales['order_counts'], bins=[0, 50, 100, 150],
                       labels=['Bronz', 'Silver','Gold'])


'''X = df[['category_id','product_id', 'unit_price', 'discount']]

y= df.quantity

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('hata :', mean_squared_error(y_test,y_pred))'''

print(df.info())
print(monthly_sales.info())
print(product_sales)
print(customer_sales)
print(df.info())  