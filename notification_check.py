import sqlite3
import os
from app import getNewsFeed
from datetime import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader

sender_email = "sentimarket@outlook.com"
subject = "Sentimarket | New stories"
smtp_server = "smtp-mail.outlook.com"
smtp_port = 587
smtp_username = "sentimarket@outlook.com"
smtp_password = "cs261project"

# Set the necessary paths and templates
script_dir = os.path.dirname(os.path.realpath(__file__))
template_path = os.path.join(script_dir, 'templates')
template_name = "email2.html"
database_path = os.path.join(script_dir, 'instance', 'cswkdb.sqlite')
# Connect to the database
connection = sqlite3.connect(database_path)
cursor = connection.cursor()

#checks for unread articles every minute.
while True:
    user_tickers = {}
    # Get all users
    select_query = "SELECT id FROM Users;"
    cursor.execute(select_query)
    rows = cursor.fetchall()
    # Loop through all stored users
    for id in rows:
        # Get the companies followed by each individual user
        user_tickers[id[0]] = []
        ticker_query = "SELECT Company.ticker from UserFollow INNER JOIN Company ON Company.id = UserFollow.companyid WHERE UserFollow.userid = ?;"
        params = (id[0],)
        cursor.execute(ticker_query, params)
        rows2 = cursor.fetchall()
        # Loop through the companies followed by each user and add their tickers to the list
        for ticker in rows2:
            user_tickers[id[0]].append(ticker[0])
    for id in user_tickers:
        if len(user_tickers[id]) > 0:
            # Get the news for each company that the user follows
            NewsFeed = getNewsFeed(user_tickers[id])
            # Gets the timestamp for which the user last received new articles
            new_timestamp = int(datetime.strptime(NewsFeed[0]["time_published"], "%Y%m%dT%H%M%S").timestamp())
            timestamp_query = "SELECT timestamp from LastUpdated WHERE userid = ?;"
            params2 = (id,)
            cursor.execute(timestamp_query, params2)
            latest_timestamp = cursor.fetchall()[0][0]
            # Gets the user's email
            email_query = "SELECT email from Users WHERE id =?"
            cursor.execute(email_query, params2)
            email = cursor.fetchall()[0][0]
            # Checks if the new article has a later date of publish than the stored one
            if new_timestamp > latest_timestamp:
                # Adds the news story to the email html and packages it up as an email to be sent to the user
                print("NEW STORIES DETECTED")
                NewNews = []
                for story in NewsFeed:
                    if int(datetime.strptime(story["time_published"], "%Y%m%dT%H%M%S").timestamp()) > latest_timestamp:
                        NewNews.append(story)
                    else:
                        break
                env = Environment(loader=FileSystemLoader(template_path))
                template = env.get_template(template_name)
                template_variables = {"stories": NewNews, "storyno": len(NewNews)}
                html_content = template.render(template_variables)
                message = MIMEMultipart()
                message['From'] = sender_email
                message['To'] = email
                message['Subject'] = subject
                message.attach(MIMEText(html_content, 'html'))
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()  
                    server.login(smtp_username, smtp_password)
                    server.sendmail(sender_email, email, message.as_string())
                # Update the stored timestamp
                new_notified_timestamp = int(datetime.strptime(NewNews[0]["time_published"], "%Y%m%dT%H%M%S").timestamp())
                update_query = "UPDATE LastUpdated SET timestamp = ? WHERE userid = ?;"
                params3 = (new_notified_timestamp, id)
                cursor.execute(update_query, params3)
                connection.commit()
                print(f"NOTIFICATION EMAIL SENT TO {email}")
            else:
                print("NO NEW STORIES")
    time.sleep(60)

        

