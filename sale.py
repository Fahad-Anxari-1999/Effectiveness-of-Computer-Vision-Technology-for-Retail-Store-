import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
file_path ='superstore.xlsx'
df = pd.read_excel(file_path, sheet_name='superstore_dataset')
def main3():


    # Function to load default data
    @st.cache_data
    def load_default_data():
        return pd.read_excel(
            'superstore.xlsx',
            sheet_name='superstore_dataset',
            engine='openpyxl'
        )

    # Function to load uploaded files (supports Excel and CSV)
    def load_uploaded_file(uploaded_file):
        try:
            if uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
                st.stop()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.stop()

    # Sidebar for file upload or default dataset
    st.sidebar.title("Upload or Load Dataset")

    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ("Default Dataset", "Upload Your Own Dataset")
    )

    # Load dataset based on user input
    if data_source == "Default Dataset":
        data = load_default_data()
        st.sidebar.success("Default dataset loaded successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

        if uploaded_file is not None:
            data = load_uploaded_file(uploaded_file)
            st.sidebar.success("Dataset uploaded successfully!")
        else:
            st.sidebar.warning("Please upload a dataset to proceed.")
            st.stop()

    # Define color palettes
    default_colors = px.colors.qualitative.Plotly
    time_series_colors = px.colors.qualitative.Set2
    # Refresh Button
    if st.button("Refresh Dashboard"):
        st.experimental_set_query_params()

    # Tooltip Message
    tooltip_message = (
        "The dataset is a working process. You cannot open the Excel file directly, "
        "and no modifications can be made. You can only add data to existing columns, "
        "and you cannot change the column names."
    )
    st.markdown(
        f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    st.sidebar.title("Point of Sale Analysis")
    options = st.sidebar.radio(
        "Select Analysis Type",
        ["Overall Overview", "Sales by Product Category", "Daily & Hourly Sales Trend","Customer Sales Analytics",
        "Inventory Turnover Rate", "Profit Margin by Product and Category", "Discount Effectiveness Analysis"]
    )

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filters positioned at the top
    min_date, max_date = min(df['order_date']), max(df['order_date'])
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Display an error if the start date is after the end date
    if start_date > end_date:
        st.sidebar.error("Start Date cannot be after End Date")

    # Additional filters
    category_filter = st.sidebar.multiselect("Select Product Category", options=df['category'].unique())
    region_filter = st.sidebar.multiselect("Select Region", options=df['region'].unique())
    product_filter = st.sidebar.multiselect("Select Product", options=df['product_name'].unique())
    segment_filter = st.sidebar.multiselect("Select Segment", options=df['segment'].unique())
    subcategory_filter = st.sidebar.multiselect("Select Subcategory", options=df['subcategory'].unique())
    state_filter = st.sidebar.multiselect("Select State", options=df['state'].unique())
    city_filter = st.sidebar.multiselect("Select City", options=df['city'].unique())

    # Filter the dataset based on sidebar selections with conditional checks
    filtered_df = df[
        (df['order_date'] >= pd.to_datetime(start_date)) &
        (df['order_date'] <= pd.to_datetime(end_date)) &
        (df['category'].isin(category_filter) if category_filter else True) &
        (df['region'].isin(region_filter) if region_filter else True) &
        (df['product_name'].isin(product_filter) if product_filter else True) &
        (df['segment'].isin(segment_filter) if segment_filter else True) &
        (df['subcategory'].isin(subcategory_filter) if subcategory_filter else True) &
        (df['state'].isin(state_filter) if state_filter else True) &
        (df['city'].isin(city_filter) if city_filter else True)
    ]

    # Overall Overview
    if options == "Overall Overview":
        st.header("Overall Business Overview")

        # Overall metrics
        total_sales = filtered_df['sales'].sum()
        total_rows = len(filtered_df)
        total_profit = filtered_df['profit'].sum()
        total_discount = filtered_df['discount'].sum()
        total_quantity = filtered_df['quantity'].sum()
        avg_profit_margin = (filtered_df['profit'].sum() / filtered_df['sales'].sum()) * 100 if total_sales != 0 else 0

        # Creating a grid for the gauge charts (3 charts per row)
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_sales = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_sales,
                title={'text': "Total Sales"},
                gauge={'axis': {'range': [0, total_sales * 1.2]},
                    'bar': {'color': "darkblue"}}
            ))
            fig_sales.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_sales, use_container_width=True)

        with col2:
            fig_profit = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_profit,
                title={'text': "Total Profit"},
                gauge={'axis': {'range': [0, total_profit * 1.2]},
                    'bar': {'color': "green"}}
            ))
            fig_profit.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_profit, use_container_width=True)

        with col3:
            fig_quantity = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_quantity,
                title={'text': "Total Quantity Sold"},
                gauge={'axis': {'range': [0, total_quantity * 1.2]},
                    'bar': {'color': "purple"}}
            ))
            fig_quantity.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_quantity, use_container_width=True)


        # Second row of metrics
        col4, col5, col6 = st.columns(3)
        with col4:
            fig_margin = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_profit_margin,
                title={'text': "Average Profit Margin (%)"},
                gauge={'axis': {'range': [0, avg_profit_margin * 1.2]},
                    'bar': {'color': "red"}}
            ))
            fig_margin.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_margin, use_container_width=True)
        # First Plot: Total Sales by Region


        with col5:
            fig_rows = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_rows,
                title={'text': "Total Number of Rows"},
                gauge={'axis': {'range': [0, total_rows * 1.2]},
                    'bar': {'color': "teal"}}
            ))
            fig_rows.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_rows, use_container_width=True)



        # First Plot: Total Sales by Region
        st.subheader("Total Sales by Region")

        # Aggregate total sales by region
        total_sales_by_region = df.groupby('region')['sales'].sum().reset_index()

        # Create a Plotly bar chart for total sales by region
        fig1 = px.bar(total_sales_by_region,
                    x='region',
                    y='sales',
                    title='Total Sales by Region',
                    labels={'region': 'Region', 'sales': 'Total Sales'},
                    color='region',  # Color by region for better distinction
                    color_discrete_sequence=px.colors.qualitative.T10)

        # Customize layout with transparent background
        fig1.update_layout(
            xaxis_title='Region',
            yaxis_title='Total Sales',
            title_x=0.5,
            template='plotly_white',
            width=700,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent overall background
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig1)

        # Second Plot: Average Profit Margin by Region
        st.subheader("Average Profit Margin by Region")

        # Calculate average profit margin by region
        avg_profit_margin_by_region = df.groupby('region')['profit_margin'].mean().reset_index()

        # Create a Plotly bar chart for average profit margin by region
        fig2 = px.bar(avg_profit_margin_by_region,
                    x='region',
                    y='profit_margin',
                    title='Average Profit Margin by Region',
                    labels={'region': 'Region', 'profit_margin': 'Average Profit Margin'},
                    color='region',  # Color by region for better distinction
                    color_discrete_sequence=px.colors.qualitative.T10)

        # Customize layout with transparent background
        fig2.update_layout(
            xaxis_title='Region',
            yaxis_title='Average Profit Margin',
            title_x=0.5,
            template='plotly_white',
            width=700,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent overall background
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig2)

        # Display first or last 5 rows of the data as a sample
        sample_data = st.radio("View Data Sample", ["First 5 rows", "Last 5 rows"])
        if sample_data == "First 5 rows":
            st.dataframe(filtered_df.head())
        else:
            st.dataframe(filtered_df.tail())

    # Sales by Product Category
    elif options == "Sales by Product Category":
        # Product Category Analysis Charts
        st.header("Sales and Profit Analysis by Product Category")

        # Aggregate data for sales and profit by product category
        category_sales_profit = filtered_df.groupby('category').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()


        # Group data by 'category' and 'product_name' to calculate aggregate metrics
        product_category_margin = filtered_df.groupby(['category', 'product_name']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()

        # Bar Chart: Total Sales by Product Category
        fig_sales_bar = px.bar(
            product_category_margin,
            x='category',
            y='sales',
            color='category',
            title="Total Sales by Product Category",
            labels={'category': 'Product Category', 'sales': 'Total Sales'},
            hover_name='product_name'
        )
        fig_sales_bar.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Total Sales',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_sales_bar)



        # Combined Chart: Scatter plot for comparing Sales and Profit
        fig_combined = px.scatter(
            category_sales_profit,
            x='sales',
            y='profit',
            text='category',
            title='Sales vs. Profit by Product Category',
            labels={'sales': 'Total Sales', 'profit': 'Total Profit'},
            color='category',
            size='sales',
            size_max=20,
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_combined.update_traces(textposition='top center')
        fig_combined.update_layout(
            xaxis_title='Total Sales',
            yaxis_title='Total Profit',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_combined)

        # Ensure 'order_date' is in datetime format before extracting year and month
        filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'], errors='coerce')

        # Now extract the year and month
        filtered_df['year'] = filtered_df['order_date'].dt.year
        filtered_df['month'] = filtered_df['order_date'].dt.month

        # Proceed with aggregating data for yearly sales and profit by product category
        yearly_category_sales_profit = filtered_df.groupby(['year', 'category']).agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()

        # Generate the charts (the code for the charts remains the same as before)
        # Chart 1: Yearly Sales by Product Category
        fig_yearly_sales = px.line(
            yearly_category_sales_profit,
            x='year',
            y='sales',
            color='category',
            title='Yearly Sales by Product Category',
            labels={'year': 'Year', 'sales': 'Total Sales'},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_yearly_sales.update_layout(
            xaxis_title='Year',
            yaxis_title='Total Sales',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_yearly_sales)



    # Daily & Hourly Sales Trend
    elif options == "Daily & Hourly Sales Trend":
        st.header("Daily and Hourly Sales Trend")

        # Select visualization level (day-wise or hour-wise)
        time_visualization = st.radio("Select Time-based Visualization", ("Day-wise", "Hour-wise"))

        # Total sales calculation
        total_sales = filtered_df['sales'].sum()
        st.subheader(f"Total Sales: ${total_sales:,.2f}")

        # If no data available
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            if time_visualization == "Day-wise":
                # Day-wise Sales
                filtered_df['day'] = filtered_df['order_date'].dt.date
                sales_over_time = filtered_df.groupby('day')['sales'].sum().reset_index()

                fig_time = px.line(
                    sales_over_time,
                    x='day',
                    y='sales',
                    title="Sales Over Time (Day-wise)",
                    markers=True,
                    color_discrete_sequence=["#FF5733"]
                )
                fig_time.update_traces(line=dict(width=2.5))
                fig_time.update_layout(xaxis_title="Date", yaxis_title="Sales", template="plotly_dark")

            else:
                # Hour-wise Sales
                filtered_df['hour'] = filtered_df['order_date'].dt.hour
                selected_hours = st.sidebar.multiselect(
                    "Select Hours", options=sorted(filtered_df['hour'].unique()),
                    default=sorted(filtered_df['hour'].unique())
                )

                # Filter the dataset by selected hours
                if selected_hours:
                    filtered_df = filtered_df[filtered_df['hour'].isin(selected_hours)]

                # Calculate total sales again after hour filter
                total_sales_hour = filtered_df['sales'].sum()
                st.subheader(f"Total Sales for Selected Hours: ${total_sales_hour:,.2f}")

                # Group data by hour for the line chart
                sales_over_time = filtered_df.groupby('hour')['sales'].sum().reset_index()

                fig_time = px.line(
                    sales_over_time,
                    x='hour',
                    y='sales',
                    title="Sales Over Time (Hour-wise)",
                    markers=True,
                    color_discrete_sequence=["#1E90FF"]
                )
                fig_time.update_traces(line=dict(width=2.5))
                fig_time.update_layout(xaxis_title="Hour", yaxis_title="Sales", template="plotly_dark")

            # Display the line chart
            st.plotly_chart(fig_time)

    elif options=="Customer Sales Analytics":
        st.header("Customer Sales Analytics")

        # Show Total Number of Customers
        total_customers = df['customer'].nunique()
        st.subheader(f"Total Number of Customers: {total_customers}")

        # Display top 5 customers by profit
        st.subheader("Top 5 Customers by Profit")
        top_customers = df.groupby('customer')['profit'].sum().nlargest(5).reset_index()
        st.dataframe(top_customers)
        if filtered_df.empty:
            st.warning("No data available for the selected date range.")
        else:
            # Select a customer to filter data
            selected_customer = st.selectbox("Select Customer", options=filtered_df['customer'].unique())
            customer_data = filtered_df.loc[filtered_df['customer'] == selected_customer].copy()

            st.subheader(f"Sales for Customer: {selected_customer}")

            # Display table for customer purchase details
            st.write("Purchase Details")
            st.dataframe(customer_data[['order_date', 'product_name', 'sales', 'quantity']])

            # Visualize sales by product for this customer
            product_sales = customer_data.groupby('product_name')['sales'].sum().reset_index()
            fig = px.bar(product_sales, y='product_name', x='sales', title=f'Sales by Product for {selected_customer}')
            st.plotly_chart(fig)

            # Visualize purchase history over time for this customer
            sales_over_time = customer_data.groupby('order_date')['sales'].sum().reset_index()
            fig = px.line(sales_over_time, x='order_date', y='sales', title=f'Sales Over Time for {selected_customer}',
                        markers=True)
            st.plotly_chart(fig)

    # Inventory Turnover Rate (ITR)
    elif options == "Inventory Turnover Rate":

        # Radio button to select Top or Bottom view
        view_type = st.radio("Select View Type:", options=["Top 5", "Bottom 5"])

        # Radio button to select the metric to sort by
        sort_metric = st.radio("Sort by:", options=["sales", "profit", "quantity"])

        # Determine if we should show the top or bottom 5 based on selected metric
        if view_type == "Top 5":
            product_table = filtered_df.nlargest(5, sort_metric)[
                ['category', 'product_name', 'sales', 'profit', 'quantity']]
        else:
            product_table = filtered_df.nsmallest(5, sort_metric)[
                ['category', 'product_name', 'sales', 'profit', 'quantity']]

        # Display the resulting table
        st.subheader(f"{view_type} Products by {sort_metric.capitalize()}")
        st.write(product_table)




        # 2. Inventory Turnover Rate Analysis
        if options == "Inventory Turnover Rate":
            st.header("Inventory Turnover Rate Analysis")

            # Calculate inventory turnover rate by category
            category_turnover = filtered_df.groupby('category').agg({
                'sales': 'sum',
                'profit': 'sum',
                'quantity': 'sum'
            }).reset_index()
            category_turnover['turnover_rate'] = category_turnover['sales'] / category_turnover['quantity']

            # Bar chart for Inventory Turnover Rate by Product Category
            fig_turnover = px.bar(
                category_turnover,
                x='category',
                y='turnover_rate',
                title='Inventory Turnover Rate by Product Category',
                labels={'category': 'Product Category', 'turnover_rate': 'Inventory Turnover Rate'},
                color='category'
            )
            st.plotly_chart(fig_turnover)



            # Quality (Quantity) vs. Sales/Profit by Category
            fig_quality_sales = px.scatter(
                category_turnover,
                x='quantity',
                y='sales',
                color='category',
                size='sales',
                title="Quality (Quantity) vs Sales by Product Category",
                labels={'quantity': 'Quality (Quantity)', 'sales': 'Total Sales'},
            )
            st.plotly_chart(fig_quality_sales)

            fig_quality_profit = px.scatter(
                category_turnover,
                x='quantity',
                y='profit',
                color='category',
                size='profit',
                title="Quality (Quantity) vs Profit by Product Category",
                labels={'quantity': 'Quality (Quantity)', 'profit': 'Total Profit'},
            )
            st.plotly_chart(fig_quality_profit)


    # Profit Margin by Product and Category
    elif options == "Profit Margin by Product and Category":


        st.header("Profit Margin Analysis by Product and Category")

        # Radio button to toggle between Top and Bottom 5 products by Profit Margin
        view_type = st.radio("Select View Type:", options=["Top 5", "Bottom 5"])

        # Group data by 'category' and 'product_name' to calculate aggregate metrics
        product_category_margin = filtered_df.groupby(['category', 'product_name']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()

        # Determine top or bottom 5 products based on profit margin
        if view_type == "Top 5":
            top_bottom_products = product_category_margin.nlargest(5, 'profit_margin')[
                ['category', 'product_name', 'sales', 'profit', 'profit_margin']]
        else:
            top_bottom_products = product_category_margin.nsmallest(5, 'profit_margin')[
                ['category', 'product_name', 'sales', 'profit', 'profit_margin']]

        # Display the resulting table with category, product name, sales, profit, and profit margin
        st.subheader(f"{view_type} Products by Profit Margin")
        st.write(top_bottom_products)

        # Group data by 'category' and 'product_name' to calculate aggregate metrics
        product_category_margin = filtered_df.groupby(['category', 'product_name']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()

        # Scatter plot: Profit Margin vs Sales by Product Category
        fig_margin_sales = px.scatter(
            product_category_margin,
            x='profit_margin',
            y='sales',
            color='category',
            size=product_category_margin['sales'].abs(),  # Absolute values to avoid negative sizes
            title="Profit Margin vs Sales by Product Category",
            labels={'profit_margin': 'Profit Margin', 'sales': 'Total Sales'},
            hover_name='product_name'
        )
        fig_margin_sales.update_layout(
            xaxis_title='Profit Margin',
            yaxis_title='Total Sales',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_margin_sales)

        # Scatter plot: Profit Margin vs Profit by Product Category
        fig_margin_profit = px.scatter(
            product_category_margin,
            x='profit_margin',
            y='profit',
            color='category',
            size=product_category_margin['profit'].abs(),  # Absolute values for size
            title="Profit Margin vs Profit by Product Category",
            labels={'profit_margin': 'Profit Margin', 'profit': 'Total Profit'},
            hover_name='product_name'
        )
        fig_margin_profit.update_layout(
            xaxis_title='Profit Margin',
            yaxis_title='Total Profit',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_margin_profit)

        # Bar Chart: Total Sales by Product Category
        fig_sales_bar = px.bar(
            product_category_margin,
            x='category',
            y='sales',
            color='category',
            title="Total Sales by Product Category",
            labels={'category': 'Product Category', 'sales': 'Total Sales'},
            hover_name='product_name'
        )
        fig_sales_bar.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Total Sales',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_sales_bar)

        # Bar Chart: Total Profit by Product Category
        fig_profit_bar = px.bar(
            product_category_margin,
            x='category',
            y='profit',
            color='category',
            title="Total Profit by Product Category",
            labels={'category': 'Product Category', 'profit': 'Total Profit'},
            hover_name='product_name'
        )
        fig_profit_bar.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Total Profit',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_profit_bar)

        # Bar Chart: Average Profit Margin by Product Category
        fig_margin_bar = px.bar(
            product_category_margin,
            x='category',
            y='profit_margin',
            color='category',
            title="Average Profit Margin by Product Category",
            labels={'category': 'Product Category', 'profit_margin': 'Average Profit Margin'},
            hover_name='product_name'
        )
        fig_margin_bar.update_layout(
            xaxis_title='Product Category',
            yaxis_title='Average Profit Margin',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_margin_bar)
        # Now extract the year and month
        filtered_df['year'] = filtered_df['order_date'].dt.year
        filtered_df['month'] = filtered_df['order_date'].dt.month

        # Proceed with aggregating data for yearly sales and profit by product category
        yearly_category_sales_profit = filtered_df.groupby(['year', 'category']).agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()

        # Chart 2: Yearly Profit by Product Category
        fig_yearly_profit = px.line(
            yearly_category_sales_profit,
            x='year',
            y='profit',
            color='category',
            title='Yearly Profit by Product Category',
            labels={'year': 'Year', 'profit': 'Total Profit'},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_yearly_profit.update_layout(
            xaxis_title='Year',
            yaxis_title='Total Profit',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_yearly_profit)





    # Discount Effectiveness Analysis
    elif options == "Discount Effectiveness Analysis":
        st.header("Discount Effectiveness Analysis")

        # Show overall discount impact (if no filter is applied)
        st.write("### Overall Discount Strategy Impact on Sales and Profit")
        overall_discount_impact = df.groupby('discount')[['sales', 'profit']].sum().reset_index()

        # Show overall discount impact using a line chart
        fig_overall = px.line(overall_discount_impact, x='discount', y=['sales', 'profit'],
                            title="Overall Sales and Profit by Discount",
                            labels={'sales': 'Total Sales', 'profit': 'Total Profit'},
                            markers=True)
        fig_overall.update_traces(mode='lines+markers')
        fig_overall.update_layout(
            xaxis_title='Discount',
            yaxis_title='Amount',
            legend_title='Metrics'
        )
        # Customize colors for the lines
        fig_overall.update_traces(line=dict(color='blue'), selector=dict(name='sales'))
        fig_overall.update_traces(line=dict(color='red'), selector=dict(name='profit'))

        # Add hover data to display detailed information
        fig_overall.update_traces(
            hovertemplate='Discount: %{x}<br>Sales: %{y}<br>Profit: %{customdata[1]}<extra></extra>',
            customdata=overall_discount_impact[['discount', 'profit']].values
        )
        st.plotly_chart(fig_overall)

        filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'], errors='coerce')

        # 2. Discount vs. Profit Margin (Scatter Plot)
        fig_discount_profit_margin = px.scatter(
            filtered_df,
            x='discount',
            y='profit_margin',
            size='sales',
            color='profit_margin',
            title='Discount vs. Profit Margin',
            labels={'discount': 'Discount (%)', 'profit_margin': 'Profit Margin'},
            color_continuous_scale=px.colors.diverging.RdYlGn,
            size_max=20
        )
        fig_discount_profit_margin.update_layout(
            xaxis_title='Discount (%)',
            yaxis_title='Profit Margin',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_discount_profit_margin)

        # 3. Sales and Profit Trends by Discount Range (Box Plot)
        # Define discount ranges (bins) for grouping
        discount_bins = pd.cut(filtered_df['discount'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
                            labels=['0-10%', '10-20%', '20-30%', '30-50%', '50-100%'])
        filtered_df['discount_range'] = discount_bins

        discount_range_sales_profit = filtered_df.groupby('discount_range').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()

        fig_discount_range_sales_profit = px.bar(
            discount_range_sales_profit,
            x='discount_range',
            y=['sales', 'profit'],
            title='Sales and Profit by Discount Range',
            labels={'discount_range': 'Discount Range', 'value': 'Amount', 'variable': 'Metrics'},
            color_discrete_sequence=px.colors.qualitative.T10,
            barmode='group'
        )
        fig_discount_range_sales_profit.update_layout(
            xaxis_title='Discount Range',
            yaxis_title='Amount',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_discount_range_sales_profit)
        # Aggregate data by 'category', 'product_name', and 'discount'
        discount_analysis = filtered_df.groupby(['category', 'product_name', 'discount']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()

        # Scatter Plot: Discount vs. Profit Margin by Product Category
        fig_discount_profit_margin = px.scatter(
            discount_analysis,
            x='discount',
            y='profit_margin',
            color='category',
            size=discount_analysis['sales'].abs(),  # Absolute value for size
            title="Discount vs Profit Margin by Product Category",
            labels={'discount': 'Discount (%)', 'profit_margin': 'Profit Margin'},
            hover_name='product_name',
            size_max=20,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_discount_profit_margin.update_layout(
            xaxis_title='Discount (%)',
            yaxis_title='Profit Margin',
            title_x=0.5,
            template='plotly_dark'
        )
        st.plotly_chart(fig_discount_profit_margin)
    
