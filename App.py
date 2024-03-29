# ----- Loading key libraries
import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import squarify
import plotly.express as px

# ----- Page setting
st.set_page_config(page_title="My App", page_icon=":👥:", layout="centered")
st.image("Customer segment.jpg")
st.title("Data Science Project")
st.header("Customer Segmentation in Online Retail")
st.write("""
         #### 👩🏻‍🏫 Lecturer: Khuat Thuy Phuong ####
         #### 👥 Team: Huynh Van Tai - Tran The Lam ####
         """)
home = """

## 🏗️ **How It's Built**

* Data preprocessing (e.g., data cleaning, feature engineering).
* RFM calculation.
* Kmeans clustering to identify customer segments.

## 🎯 **Key Features**

* Predict customer segments for uploaded data (optional).
* Build customer segmentation for new dataset (optional).
* Input specific customer IDs for segment prediction.

## 🚀 **Getting Started**
1. Upload your customer data (CSV format) or enter customer IDs.
2. Click "Predict" to predict the segment for each customer.

"""

# 1. ----- Read data (outside conditional block)
@st.cache_data  # Consider allow_reload=True for updates if needed
def load_data():
    return pd.read_csv("OnlineRetail.csv", encoding='latin-1')
data = load_data()

# Convert DataFrame to bytes
csv_file = data.to_csv(index=False, encoding='utf-8')
csv_file = csv_file.encode('utf-8')  # Convert to bytes

if st.download_button(label="Download Raw Data", data=csv_file, file_name="OnlineRetail.csv", help="Click to download Raw dataset of Online Retail"):
    st.text("Data Downloaded Successfully")

# Run model
with open('kmean_model.pkl', 'rb') as file:
    model_kmeans_lds6 = pickle.load(file)
scaler = joblib.load('scaler.pkl')

import pandas as pd
def remove_outliers_iqr(df, X):
    for col in X:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers_ = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print('Column {} has {} outliers'.format(col,outliers))
        if outliers_.shape[0] > 0:
            df.drop(outliers_.index, axis=0, inplace=True)

def get_dataset_clean(data1):
    # Check if data1 is a DataFrame
    if not isinstance(data1, pd.DataFrame):
        raise ValueError("Input is not a DataFrame")
    # Drop rows with missing CustomerID
    data1.dropna(subset=['CustomerID'], inplace=True)
    # Remove rows where InvoiceNo starts with 'C'
    data1 = data1.loc[~data1["InvoiceNo"].str.startswith('C', na=False)]
    data1 = data1.drop(data1[data1['UnitPrice'] < 0].index)  
    data1 = data1.drop(data1[data1['Quantity'] < 0].index)
    input_col_num= ['Quantity', 'UnitPrice']
    remove_outliers_iqr(data1, input_col_num)
    # Get the mode of UnitPrice for each StockCode
    unitprice_mode = data1.groupby('StockCode')['UnitPrice'].apply(lambda x: x.mode().iloc[0])   
    # Replace zero or null UnitPrice with mode
    data1['UnitPrice'] = data1.apply(lambda row: unitprice_mode[row['StockCode']] if row['UnitPrice'] == 0 or pd.isnull(row['UnitPrice']) else row['UnitPrice'], axis=1)
    # Convert InvoiceDate to datetime and extract date
    data1['InvoiceDate'] = pd.to_datetime(data1['InvoiceDate'], format='%d-%m-%Y %H:%M').dt.date
    # Calculate Revenue
    data1['Revenue'] = data1['Quantity'] * data1['UnitPrice']
    return data1

def predict_segmentKmean_data(data1):
    # Caculate RFM values
    max_date = data1['InvoiceDate'].max()
    Recency = lambda x: (max_date - x.max()).days
    Frequency = lambda x: len(x.unique())
    Monetary = lambda x: round(sum(x), 2)
    rfm_values = data1.groupby('CustomerID').agg({'InvoiceDate': Recency, 'InvoiceNo': Frequency, 'Revenue': Monetary}) 
    rfm_values.columns = ['Recency', 'Frequency', 'Monetary']
    # Scale RFM values
    rfm_values_scaled = scaler.fit_transform(rfm_values)
    # Load model and predict cluster
    clusters = model_kmeans_lds6.predict(rfm_values_scaled)
    # Map cluster to segment        
    segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
    segment_name = [segments[i] for i in clusters]
    rfm_values['Cluster_Kmeans'] = segment_name
    return segment_name, rfm_values
    
# Function to predict segment using KMeans
def predict_segmentKmean(CustomerID, data1):
    if CustomerID not in data1['CustomerID'].values:
        return 'Không tìm thấy khách hàng'
    else:
        # Caculate RFM values
        max_date = data1['InvoiceDate'].max()
        Recency = lambda x: (max_date - x.max()).days
        Frequency = lambda x: len(x.unique())
        Monetary = lambda x: round(sum(x), 2)
        rfm_values = data1.groupby('CustomerID').agg({'InvoiceDate': Recency, 'InvoiceNo': Frequency, 'Revenue': Monetary}) 
        rfm_values.columns = ['Recency', 'Frequency', 'Monetary']
        # Scale RFM values
        rfm_values_scaled = scaler.fit_transform(rfm_values)
        # Load model and predict cluster
        cluster = model_kmeans_lds6.predict(rfm_values_scaled)
        # Map cluster to segment        
        segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
        segment_name = segments.get(cluster[0], 'Unknown segment')
        rfm_values['Cluster_Kmeans'] = segment_name
        return segment_name, rfm_values

def visualize_rfm_squarify(rfm_values):
    rfm_agg = rfm_values.groupby('Cluster_Kmeans').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)
    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    # Change the Cluster Columns Datatype into discrete values
    rfm_agg['Cluster_Kmeans'] = 'Cluster ' + rfm_agg['Cluster_Kmeans'].astype(str)

    colors_dict_cluster = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
               'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}
    # Creat chart
    fig1, ax = plt.subplots(figsize=(14, 10))

    squarify.plot(sizes= rfm_agg['Count'],
              text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
              color=colors_dict_cluster.values(),
              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
              for i in range(0, len(rfm_agg))], alpha=0.5 )
    plt.title("Customers Segments Kmeans", fontsize=26, fontweight="bold")
    plt.axis('off')

    fig2 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster_Kmeans",
           hover_name="Cluster_Kmeans", size_max=70, color_discrete_map=colors_dict_cluster)
    # Display the plot using Streamlit
    st.pyplot(fig1)
    st.plotly_chart(fig2)

# GUI
menu = ["🏠Home", "🛒Predict for New RFM Value", "👨‍💼Predict for CustomerID", "📈RFM For New Dataset"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == '🏠Home':    
    st.markdown(home, unsafe_allow_html=True)

elif choice == '📈RFM For New Dataset':
    st.subheader("Select data")
    df = data.copy()
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file is not None:
        st.write(f"**Predicted Segment Summary For New dataset:**")
        df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        st.write(f"**Predicted Segment Summary For Retail Online dataset:**")
    df1 = get_dataset_clean(df)
    segment_name, rfm_values = predict_segmentKmean_data(df1)
    st.write(f"**Predicted Segment Summary:**")
    # Display segment counts
    segment_counts = rfm_values['Cluster_Kmeans'].value_counts().reset_index(name='Count')
    st.write(segment_counts)
    # Visualize RFM segments
    st.subheader("RFM Segments Visualization")
    visualize_rfm_squarify(rfm_values)

elif choice == '🛒Predict for New RFM Value':
    st.subheader("Enter New RFM Values")
    data_rfm = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
    for i in range(2):
        st.write(f"Customer {i+1}")
        # Tạo các slider để nhập giá trị cho cột Recency, Frequency, Monetary
        recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
        frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
        monetary = st.slider("Monetary", 1, 10000, 100, key=f"monetary_{i}")
        new_customer_data = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}
        data_rfm = pd.concat([data_rfm, pd.DataFrame(new_customer_data, index=[0])], ignore_index=True)

    if st.button("Predict"):
        try:
            data_rfm_sca = scaler.fit_transform(data_rfm)
            cluster = model_kmeans_lds6.predict(data_rfm_sca)
            segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
            for i, segment_index in enumerate(cluster):
                segment_name = segments.get(segment_index, 'Unknown segment')
                st.write(f"**Predicted Segment {i+1}:** {segment_name}")
        except ValueError:
            st.error("Invalid input.")
                 
elif choice ==  '👨‍💼Predict for CustomerID':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload Customer data or Input data?", options=("Upload", "Input"))
    if type == "Upload":
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            df1 = pd.read_csv(uploaded_file_1, encoding='latin-1')
            df = get_dataset_clean(df1)
            st.dataframe(df)
            lines = df.iloc[:, df.columns.get_loc('CustomerID')].astype(int)
            lines = np.array(lines)
            st.code(lines)
            customer_ids_set = set()
            predictions = []
            for ids in lines:
                if ids not in customer_ids_set:
                    customer_ids_set.add(ids)
                    segment = predict_segmentKmean(ids, df)
                    predictions.append((ids, segment))
            for ids, segment in predictions:
                st.code(f"The predicted segment for Customer ID {ids} is: {segment}")

            if not customer_ids_set:
                st.error("Invalid input: ")
           
    if type == "Input":
        df = data.copy()
        df = get_dataset_clean(df)
        CustomerID = st.text_area(label="Input CustomerID (separated by commas):")
        predict_button = st.button("Predict")
        if predict_button:
            if CustomerID:
                try:
                    CustomerID = [int(id.strip()) for id in CustomerID.split(",")]
                    lines = np.array(CustomerID)
                    st.code(lines)
                    for CustomerID in lines:
                        segment = predict_segmentKmean(CustomerID, df)
                        st.code(f"The predicted segment for Customer ID {CustomerID} is: " + str(segment))
                except ValueError:
                    st.error("Invalid input: Please enter integers separated by commas.")