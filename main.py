import streamlit as st
import pandas as pd
from Foot_Traffic_Analysis import main2  # Import program1 function
from Shelf_Monitoring_and_Cheakout_Efficiency import main1  # Import program2 function
from sale import main3

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Foot Traffic Analysis", "Shelf Monitoring and Cheakout Efficiency", "Processed Videos", "Rush Hours Analysis" ,"Sales Analysis"])
file_path ='superstore.xlsx'
df = pd.read_excel(file_path, sheet_name='superstore_dataset')
# Render the selected page
if page == "Home":
    st.title("Main App")
    st.write("Welcome! Use the sidebar to navigate to the programs.")
    if st.button("Foot Traffic Analysis"):
        main2()
    elif st.button("Shelf Monitoring and Cheakout Efficiency"):
        main1()
    elif st.button("Processed Videos"):
        st.title("Foot Traffic Analysis")
        video_file_path1 = "Processed_Foot_Traffic_Analysis_Converted.mp4"
        video_file_path2 = "Processed_Shelf_Monitoring_and_Cheakout_Efficiency_Converted.mp4"


        try:
            # Open the video file in binary mode
            video_file = open(video_file_path1, "rb")
            video_bytes = video_file.read()

            # Display the video in Streamlit
            st.video(video_bytes)

        except FileNotFoundError:
            st.error(f"Video file not found at {video_file_path1}. Please check the file name and path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        st.title("Shelf Monitoring and Cheakout Efficiency")
        try:
            # Open the video file in binary mode
            video_file1 = open(video_file_path2, "rb")
            video_bytes1 = video_file1.read()

            # Display the video in Streamlit
            st.video(video_bytes1)

        except FileNotFoundError:
            st.error(f"Video file not found at {video_file_path2}. Please check the file name and path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif st.button("Rush Hours Analysis"):
        df= pd.read_csv('Shelf_Monitoring_and_Cheakout_Efficiency.csv')
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Calculate the average for each numeric column
        averages = numeric_df.mean()
        display_columns = ['Timestamp', 'Current Customers', 'Current Visitor','Shelf 1', 'Shelf 2','Shelf 3','Shelf 4' ]
        display_columns = list(dict.fromkeys(display_columns))
    # Iterate through each column and print rows above average with their timestamps
        for column in numeric_df.columns:
                # Skip Timestamp column for averaging
            # Filter rows where the current column's value is above its average
            if column in display_columns:
                continue
            above_average_rows = df[df[column] > averages[column]][display_columns + [column]]
            
            # Print results
            st.title(f"  {column} in peek Hours :")
            st.write(above_average_rows)
            print("\n")
    elif st.button("Sales Analysus"):
        main3()

elif page == "Foot Traffic Analysis":
    main2()  # Call the function from program1
elif page == "Shelf Monitoring and Cheakout Efficiency":
    main1()  # Call the function from program2
elif page =="Processed Videos":
    st.title("Foot Traffic Analysis")
    video_file_path1 = "Processed_Foot_Traffic_Analysis_Converted.mp4"
    video_file_path2 = "Processed_Shelf_Monitoring_and_Cheakout_Efficiency_Converted.mp4"


    try:
        # Open the video file in binary mode
        video_file = open(video_file_path1, "rb")
        video_bytes = video_file.read()

        # Display the video in Streamlit
        st.video(video_bytes)

    except FileNotFoundError:
        st.error(f"Video file not found at {video_file_path1}. Please check the file name and path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.title("Shelf Monitoring and Cheakout Efficiency")
    try:
        # Open the video file in binary mode
        video_file1 = open(video_file_path2, "rb")
        video_bytes1 = video_file1.read()

        # Display the video in Streamlit
        st.video(video_bytes1)

    except FileNotFoundError:
        st.error(f"Video file not found at {video_file_path2}. Please check the file name and path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
elif page =="Rush Hours Analysis":
    df= pd.read_csv('Shelf_Monitoring_and_Cheakout_Efficiency.csv')
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the average for each numeric column
    averages = numeric_df.mean()
    display_columns = ['Timestamp', 'Current Customers', 'Current Visitor','Shelf 1', 'Shelf 2','Shelf 3','Shelf 4' ]
    display_columns = list(dict.fromkeys(display_columns))
# Iterate through each column and print rows above average with their timestamps
    for column in numeric_df.columns:
             # Skip Timestamp column for averaging
        # Filter rows where the current column's value is above its average
        if column in display_columns:
            continue
        above_average_rows = df[df[column] > averages[column]][display_columns + [column]]
        
        # Print results
        st.title(f"  {column} in peek Hours :")
        st.write(above_average_rows)
        print("\n")
elif page == "Sales Analysis":
    main3() 
