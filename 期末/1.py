import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import joblib
# -------------------- 数据加载与初始化 --------------------
@st.cache_data
def load_data():
    df = pd.read_csv('学生数据.csv',encoding='UTF-8') # 读取数据文件路径
    return df
df = load_data()
# 初始化模型（若需训练新模型，可取消注释下方训练代码）
# def train_model():
# X = df[["每周学习时长", "上课出勤率", "期中考试分数", "作业完成率"]]
# y = df["期末考试分数"]
# model = LinearRegression()
# model.fit(X, y)
# joblib.dump(model, "score_predictor.pkl")
# train_model() # 首次运行时训练模型，之后可注释
model = joblib.load("score_predictor.pkl") # 加载训练好的模型


import streamlit as st
st.set_page_config(
    page_title="学生成绩分析与预测系统",page_icon='🎓',layout='wide',)

with st.sidebar:
    st.title('🎓导航菜单')
    page = st.radio(
        "选择页面",
        ("项目分析","专业数据介绍",  "成绩预测")
    )

if page == "项目分析":
    st.title("🎓学生成绩分析与预测系统")
    c1,c2=st.columns([1,1])
    with c1:
        st.header('📝项目概括')
        st.text('本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩')
        st.subheader('主要特点')
        st.markdown('**📊数据可视化:** 多维展示学生学业数据')
        st.markdown('**🎯专业分析:** 按专业分类的详细统计分析')
        st.markdown('**☸智能预测:** 基于机器学习模型的成绩预测')
        st.markdown('**💡学习建议:** 根据预测结果提供个性化反馈')
    with c2:
        st.image('学生数据分析示意图.png',caption='学生数据分析示意图',width=800)

    st.header('🚀项目目标')
    a1,a2,a3=st.columns(3)
    
    with a1:        
        st.subheader('🎯目标一')
        st.markdown('**分析影响因素**')
        st.text('· 识别关键学习指标')
        st.text('· 识别关键学习指标')
        st.text('· 识别关键学习指标')
    with a2:
        st.subheader('📠目标二 ')
        st.markdown('**可视化展示**')
        st.text('· 专业对比分析')
        st.text('· 性别差异研究')
        st.text('· 学习模式识别')
    with a3:
        st.subheader('🔮目标三 ')
        st.markdown('**成绩预测**')
        st.text('· 机器学习模型')
        st.text('· 个性化预测')
        st.text('· 及时干预预警')

    st.header('🛠技术架构')
    b1,b2,b3=st.columns(3)
    
    with b1:        
        st.markdown('**前端框架**')
        python_code = ''' Streamlit
    '''
        st.code(python_code, language=None)
    with b2:
        st.markdown('**数据处理**')
        python_code1 = ''' Pandas
        Numpy
    '''
        st.code(python_code1, language=None)
        
    with b3:
        st.markdown('**可视化**')
        python_code2 = ''' plotly
    Matplotlib
    '''
        st.code(python_code2, language=None)


elif page =="专业数据介绍":   
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    def show_major_analysis():
        st.title("📊专业数据分析")

        # 加载数据集
        df = pd.read_csv('学生数据.csv')

        # （1）各专业男女性别比例
        st.subheader("1. 各专业男女性别比例")
        gender_ratio = df.groupby(['专业', '性别']).size().unstack(fill_value=0)
        gender_ratio['总人数'] = gender_ratio['男'] + gender_ratio['女']
        gender_ratio['男性比例'] = gender_ratio['男'] / gender_ratio['总人数']
        gender_ratio['女性比例'] = gender_ratio['女'] / gender_ratio['总人数']
        fig1 = px.bar(gender_ratio.reset_index(), x='专业', y=['男性比例', '女性比例'], barmode='group', labels={'value': '比例'})
        fig1.update_layout(legend_title='性别')
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.dataframe(
                gender_ratio[['男性比例', '女性比例']].reset_index()
                .rename(columns={'男性比例': '男', '女性比例': '女'})
                .set_index('专业'), 
                use_container_width=True
            )

        # （2）各专业学习指标对比
        st.subheader("2. 各专业学习指标对比")
        score_trend = df.groupby('专业').agg({
            '期中考试分数': 'mean', 
            '期末考试分数': 'mean', 
            '每周学习时长（小时）': 'mean'
        }).reset_index()
        score_trend_melt = pd.melt(score_trend, id_vars='专业', value_vars=['期中考试分数', '期末考试分数'], var_name='考试类型', value_name='分数')
        fig2 = px.line(score_trend_melt, x='专业', y='分数', color='考试类型', title='各专业期中期末成绩趋势')
        fig2.add_scatter(x=score_trend['专业'], y=score_trend['每周学习时长（小时）'], name='每周学习时长', yaxis='y2')
        fig2.update_layout(yaxis2=dict(title='每周学习时长（小时）', overlaying='y', side='right'))
        col3, col4 = st.columns([2, 1])
        with col3:
            st.plotly_chart(fig2, use_container_width=True)
        with col4:
            st.dataframe(score_trend.set_index('专业'), use_container_width=True)

        # （3）各专业出勤率分析
        st.subheader("3. 各专业出勤率分析")
        attendance = df.groupby('专业')['上课出勤率'].mean().reset_index()
        # 改用px.bar的color参数直接映射数值，避免coloraxis报错
        fig3 = px.bar(
            attendance, 
            x='专业', 
            y='上课出勤率', 
            labels={'上课出勤率': '平均上课出勤率'},
            color='上课出勤率',  # 直接用color映射数值
            color_continuous_scale='Viridis',  # 颜色刻度
            range_color=[0, 1]  # 颜色范围
        )
        col5, col6 = st.columns([2, 1])
        with col5:
            st.plotly_chart(fig3, use_container_width=True)
        with col6:
            attendance_rank = attendance.sort_values('上课出勤率', ascending=False).reset_index(drop=True)
            attendance_rank.index += 1
            st.dataframe(
                attendance_rank.rename(columns={'专业': '专业', '上课出勤率': '平均出勤率'})
                .set_index('专业'), 
                use_container_width=True
            )

        # （4）大数据管理专业专项分析
        st.subheader("4. 大数据管理专业专项分析")
        big_data = df[df['专业'] == '大数据管理']
        avg_attendance = big_data['上课出勤率'].mean()
        avg_final_score = big_data['期末考试分数'].mean()
        avg_study_hours = big_data['每周学习时长（小时）'].mean()
        pass_rate = (big_data['期末考试分数'] >= 60).mean()
        col7, col8, col9, col10 = st.columns(4)
        with col7:
            st.metric("平均出勤率", f"{avg_attendance:.1%}")
        with col8:
            st.metric("平均期末分数", f"{avg_final_score:.1f}分")
        with col9:
            st.metric("通过率", f"{pass_rate:.1%}")
        with col10:
            st.metric("平均学习时长", f"{avg_study_hours:.1f}小时")
        fig4 = px.histogram(big_data, x='期末考试分数', nbins=20, title='大数据管理专业期末成绩分布')
        fig5 = px.box(big_data, y='每周学习时长（小时）', title='大数据管理专业学习时长分布')
        col11, col12 = st.columns(2)
        with col11:
            st.plotly_chart(fig4, use_container_width=True)
        with col12:
            st.plotly_chart(fig5, use_container_width=True)
    show_major_analysis()

else:
    st.title("🔮期末成绩预测")

    with st.form("predict_form"):
        st.subheader("请输入学生信息")
        student_id = st.text_input("学号")
        gender = st.selectbox("性别", ["男", "女"])
        major = st.selectbox("专业", df["专业"].unique())
        study_hours = st.number_input("每周学习时长（小时）", min_value=0.0, max_value=50.0, step=0.1)
        attendance = st.number_input("上课出勤率", min_value=0.0, max_value=1.0, step=0.01)
        mid_score = st.number_input("期中考试分数", min_value=0.0, max_value=100.0, step=0.1)
        homework_rate = st.number_input("作业完成率", min_value=0.0, max_value=1.0, step=0.01)
        submit = st.form_submit_button("预测成绩")

    if submit:

        X = [[study_hours, attendance, mid_score, homework_rate]]
        pred_score = model.predict(X)[0]
        pred_score = max(0, min(100, pred_score)) 
        st.subheader("📊预测结果")
        st.markdown(f"**预测期末成绩：{pred_score:.2f} 分**")
        if pred_score >= 80:
            st.image("https://pic2.zhimg.com/v2-540f5061894291e7a0d2aa7fc6c23471_b.jpg") 
        elif pred_score >= 60:
            st.success("成绩合格，继续保持！")
            st.image('https://static.aipiaxi.com/image/2023/11/FosT1Eppd_hmbPu6dFQGBhxnF92E.jpeg')
        else:
            st.warning("成绩待提高，建议加强学习！")
            st.image('https://img.dancihu.com/pic/2023-07-18/c1daf2cb-a712-b2e7-ccc5-ecd3b307d324.jpeg')
        



        


    
       
        
    
