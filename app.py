import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="My Tabbed App",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("A visual analytics tool for explainable large VLMs")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generation", "Attention", "Hidden Representations", "Misc."])

# Tab 1: Data Analysis
with tab1:
    st.header("Data Analysis")
    
    # Generate sample data
    @st.cache_data
    def load_sample_data():
        dates = pd.date_range('2023-01-01', periods=100)
        data = {
            'Date': dates,
            'Sales': np.random.normal(1000, 200, 100),
            'Customers': np.random.poisson(50, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        }
        return pd.DataFrame(data)
    
    df = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Sales'])
        ax.set_title('Daily Sales Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Sales by Category")
        category_sales = df.groupby('Category')['Sales'].sum()
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
        ax2.set_title('Sales Distribution by Category')
        st.pyplot(fig2)
    
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

# Tab 2: Form Input
with tab2:
    st.header("User Input Form")
    
    with st.form("user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
        
        with col2:
            department = st.selectbox("Department", 
                                    ["Sales", "Marketing", "Engineering", "HR"])
            experience = st.slider("Years of Experience", 0, 40, 5)
            subscribe = st.checkbox("Subscribe to newsletter")
        
        comments = st.text_area("Additional Comments")
        
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            st.success("Form submitted successfully!")
            st.write("**Submitted Information:**")
            st.write(f"- Name: {name}")
            st.write(f"- Email: {email}")
            st.write(f"- Age: {age}")
            st.write(f"- Department: {department}")
            st.write(f"- Experience: {experience} years")
            st.write(f"- Newsletter: {'Yes' if subscribe else 'No'}")
            if comments:
                st.write(f"- Comments: {comments}")

# Tab 3: Maps
with tab3:
    st.header("Interactive Maps")
    
    # Generate sample location data
    @st.cache_data
    def generate_map_data():
        return pd.DataFrame({
            'lat': np.random.normal(37.7749, 0.1, 100),
            'lon': np.random.normal(-122.4194, 0.1, 100),
            'size': np.random.randint(20, 100, 100),
            'color': np.random.choice(['red', 'blue', 'green', 'orange'], 100)
        })
    
    map_data = generate_map_data()
    
    st.subheader("Sample Locations Map")
    st.map(map_data[['lat', 'lon']])
    
    st.subheader("Map Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        zoom_level = st.slider("Zoom Level", 1, 20, 10)
        show_labels = st.checkbox("Show Labels", value=True)
    
    with col2:
        map_style = st.selectbox("Map Style", 
                                ["Default", "Satellite", "Terrain"])
        point_size = st.slider("Point Size", 1, 10, 5)

# Tab 4: Settings
with tab4:
    st.header("Application Settings")
    
    st.subheader("Display Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
    
    with col2:
        timezone = st.selectbox("Timezone", 
                               ["UTC", "EST", "PST", "GMT", "CET"])
        date_format = st.selectbox("Date Format", 
                                  ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
    
    st.subheader("Notifications")
    email_notifications = st.checkbox("Email Notifications", value=True)
    push_notifications = st.checkbox("Push Notifications", value=False)
    weekly_reports = st.checkbox("Weekly Reports", value=True)
    
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        auto_save = st.checkbox("Auto-save", value=True)
        backup_frequency = st.selectbox("Backup Frequency", 
                                       ["Daily", "Weekly", "Monthly"])
    
    with col2:
        data_retention = st.number_input("Data Retention (days)", 
                                        min_value=1, max_value=365, value=90)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        st.balloons()

# Sidebar (optional)
with st.sidebar:
    st.header("Navigation")
    st.write("Use the tabs above to navigate between different sections of the application.")
    
    st.header("Quick Stats")
    st.metric("Total Users", "1,234", "12")
    st.metric("Active Sessions", "56", "-2")
    st.metric("Revenue", "$12,345", "8%")