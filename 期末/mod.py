import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. 加载数据
df = pd.read_csv('学生数据.csv',encoding='UTF-8') # 读取数据文件路径

# 2. 定义特征和目标变量
features = ["每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率"]
X = df[features]
y = df["期末考试分数"]

# 3. 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 4. 保存模型
joblib.dump(model, "score_predictor.pkl")

print("模型训练完成并保存为 score_predictor.pkl")
