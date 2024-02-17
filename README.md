# LinkedIn-Scraper_analysis
The goal of this project was to scrape data from LinkedIn using Selenium web driver and Beautiful Soup, clean the data in Excel, perform SQL queries on the data, and create a dashboard in Excel.

Process:
1.	Scraping data from LinkedIn: The Selenium web driver was used to navigate to the LinkedIn website and search for profiles with specific criteria. Beautiful 
           Soup was then used to parse the HTML code of the profiles and extract the relevant data.
  	 #### LinkedIn Data Scrapping code

3.	Collecting data in Excel: The scraped data was imported into Excel file.
     #### linkedin_data 2023-03-31_final.csv
   
4.  Data Cleaning in Python Notebook: Excel file was imported into the python notebook and Cleaned to remove any unnecessary information. This included 
          removing duplicates and formatting the data to ensure that it was easy to work with.
     #### Project 2_1.ipynb

6.  Data Visualization: The cleaned and queried data was then used to create a dashboard in Excel.
      This included creating charts and graphs to visualize the data and presenting it in a clear and concise manner.

7.	Model Building: Recommendation Function were formed to collect data according to State, Region also Job_name

8.	Deployment: Streamlit library was used for deployment of linkedIn data to obtain top 10 statewise jobs or regionwise jobs or jobwise region.
     #### deploy_file.py

Results:
The scraping, cleaning, modelling, and deployment processes were successful in extracting, organizing, and presenting the data in a meaningful way.
 The resulting deployment provided valuable insights into the data and allowed for easy analysis and interpretation.


Conclusion:
Overall, the project was a success in using Selenium web driver, Beautiful Soup, Excel, and python notebook cleaning, visualization, modelling,
deploying, and analyze data from LinkedIn. The resulting dashboard provided a valuable data for understanding and interpreting the data.
