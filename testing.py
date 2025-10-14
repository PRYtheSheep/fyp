import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="FYP Visual analytics tool",
    page_icon="📊",
    layout="wide"
)

st.title("Interactive Graph with Image on Hover")
st.markdown("Click on a point to see its image, or use the interactive hover display below")

# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Sidebar for data input
st.sidebar.header("Configure Your Data")

# Choose between upload or URL
input_method = st.sidebar.radio("Image Input Method:", ["Upload Images", "Use Image URLs"])

# Number of data points
num_points = st.sidebar.number_input("Number of data points:", min_value=1, max_value=20, value=5)

# Create input fields for each point
st.sidebar.markdown("---")
st.sidebar.subheader("Enter Data Points")

datapoints = []
for i in range(num_points):
    st.sidebar.markdown(f"**Point {i+1}**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        x = st.number_input(f"X", key=f"x_{i}", value=float(i+1))
        y = st.number_input(f"Y", key=f"y_{i}", value=float((i+1)*2))
    
    name = st.sidebar.text_input(f"Name", key=f"name_{i}", value=f"Point {i+1}")
    
    # Image input based on method
    if input_method == "Upload Images":
        uploaded_file = st.sidebar.file_uploader(
            f"Image for {name}", 
            type=['png', 'jpg', 'jpeg', 'gif'],
            key=f"img_{i}"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            # Resize for performance
            img.thumbnail((400, 400))
            image_url = image_to_base64(img)
        else:
            # Default placeholder
            image_url = f"https://via.placeholder.com/300x200/{'%06x' % (hash(name) % 0xFFFFFF)}/FFFFFF?text={name.replace(' ', '+')}"
    else:
        image_url = st.sidebar.text_input(
            f"Image URL", 
            key=f"url_{i}",
            value=f"https://via.placeholder.com/300x200/{'%06x' % (hash(name) % 0xFFFFFF)}/FFFFFF?text={name.replace(' ', '+')}"
        )
    
    datapoints.append({
        'x': x,
        'y': y,
        'name': name,
        'image_url': image_url
    })
    
    st.sidebar.markdown("---")

# Create DataFrame
df = pd.DataFrame(datapoints)

# Customization options
st.sidebar.markdown("### Chart Customization")
marker_size = st.sidebar.slider("Marker Size:", 5, 30, 15)
marker_color = st.sidebar.color_picker("Marker Color:", "#1f77b4")

# Create the plot with click events
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=marker_size, 
        color=marker_color,
        line=dict(width=2, color='white')
    ),
    text=df['name'],
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x}<br>' +
                  'Y: %{y}<br>' +
                  '<i>Click to view image</i>' +
                  '<extra></extra>'
))

fig.update_layout(
    title='Click on points to see images below',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    hovermode='closest',
    height=500,
    plot_bgcolor='rgba(240,240,240,0.5)'
)

# Create side-by-side layout for graph and image
col_graph, col_image = st.columns([1.5, 1])

with col_graph:
    # Display the chart with click events
    clicked_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart")

with col_image:
    st.markdown("### 📍 Selected Point")
    
    if clicked_point and clicked_point.selection and clicked_point.selection.points:
        # Get the clicked point index
        point_indices = [p['point_index'] for p in clicked_point.selection.points]
        
        if point_indices:
            idx = point_indices[0]
            selected_data = df.iloc[idx]
            
            st.markdown(f"**{selected_data['name']}**")
            st.metric("X Coordinate", f"{selected_data['x']:.2f}")
            st.metric("Y Coordinate", f"{selected_data['y']:.2f}")
            st.image(selected_data['image_url'], caption=selected_data['name'], use_column_width=True)
    else:
        st.info("👈 Click on any point in the graph to see its image")

# Alternative: Hover-based display using Streamlit columns
st.markdown("---")
st.markdown("### 🖱️ Browse All Images")

# Create a grid of images with hover effect
cols_per_row = 3
rows = [df.iloc[i:i+cols_per_row] for i in range(0, len(df), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for idx, (col, (_, point)) in enumerate(zip(cols, row.iterrows())):
        with col:
            st.image(point['image_url'], use_column_width=True)
            st.caption(f"**{point['name']}** (X: {point['x']:.1f}, Y: {point['y']:.1f})")

# Show data table
with st.expander("📊 View Data Table"):
    st.dataframe(df[['name', 'x', 'y']], use_container_width=True)

# Export option
st.markdown("---")
if st.button("📥 Download Data as CSV"):
    csv = df[['name', 'x', 'y']].to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="plotly_data.csv",
        mime="text/csv"
    )

# Tips
with st.expander("💡 How It Works"):
    st.markdown("""
    **Why doesn't Plotly show images on hover?**
    - Plotly's standard hover tooltips don't support embedded images
    - HTML img tags aren't rendered in hovertemplate
    
    **Solutions in this app:**
    1. **Click to view**: Click any point to see its image below the chart
    2. **Gallery view**: Browse all images in the grid below
    3. **Interactive selection**: Use Plotly's built-in selection to view details
    
    **Alternative approaches:**
    - Use a JavaScript callback (requires custom HTML/JS)
    - Use Plotly Dash for more interactive features
    - Create custom tooltips with separate overlay divs
    
    **Best practice:**
    For image display on interaction, click-based or selection-based approaches 
    work better than hover in Streamlit + Plotly combinations.
    """)